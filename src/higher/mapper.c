/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *******************************************************************************/

#include "aml.h"

#include "aml/higher/mapper.h"
#include "aml/layout/dense.h"
#include "aml/utils/inner-malloc.h"
#include "aml/utils/queue.h"

#define PTR_OFF(ptr, sign, off) (void *)((intptr_t)(ptr)sign(intptr_t)(off))

int aml_mapper_create(struct aml_mapper **out,
                      uint64_t flags,
                      const size_t struct_size,
                      const size_t num_fields,
                      const size_t *fields_offset,
                      const num_element_fn *num_elements,
                      struct aml_mapper **fields)
{
	if (num_fields > 0 && (fields_offset == NULL || fields == NULL))
		return -AML_EINVAL;

	struct aml_mapper *m;
	size_t extra = num_fields * sizeof(struct aml_mapper *) +
	               (num_elements ? num_fields * sizeof(num_element_fn) : 0);

	if (num_fields == 0) {
		m = AML_INNER_MALLOC(struct aml_mapper);
		if (m == NULL)
			return -AML_ENOMEM;
		m->size = struct_size;
		m->n_fields = 0;
		m->offsets = NULL;
		m->num_elements = NULL;
		m->fields = NULL;
	} else {
		m = AML_INNER_MALLOC_EXTRA(num_fields, size_t, extra,
		                           struct aml_mapper);
		if (m == NULL)
			return -AML_ENOMEM;

		m->size = struct_size;
		m->n_fields = num_fields;
		m->offsets = AML_INNER_MALLOC_GET_ARRAY(m, size_t,
		                                        struct aml_mapper);
		memcpy(m->offsets, fields_offset,
		       num_fields * sizeof(*fields_offset));

		m->fields = AML_INNER_MALLOC_GET_EXTRA(m, num_fields, size_t,
		                                       struct aml_mapper);
		memcpy(m->fields, fields, num_fields * sizeof(*fields));

		if (num_elements == NULL)
			m->num_elements = NULL;
		else {
			m->num_elements =
			        (num_element_fn *)((char *)m->fields +
			                           num_fields *
			                                   sizeof(*fields));
			memcpy(m->num_elements, num_elements,
			       num_fields * sizeof(*num_elements));
		}
	}
	m->flags = flags;
	*out = m;
	return AML_SUCCESS;
}

void aml_mapper_destroy(struct aml_mapper **mapper)
{
	if (mapper) {
		if (*mapper)
			free(*mapper);
		*mapper = NULL;
	}
}

#define get_num_elements(m, i, ptr)                                            \
	((m->num_elements && m->num_elements[i]) ? m->num_elements[i](ptr) : 1)

// Helper function to copy src into dst.
static int aml_mapper_memcpy(void *src,
                             void *dst,
                             size_t size,
                             struct aml_dma *dma,
                             aml_dma_operator dma_op,
                             void *dma_op_arg)
{
	int err;
	struct aml_layout *_src, *_dst;
	size_t dims = 1, stride = 0, pitch = size;

	// Create layouts
	err = aml_layout_dense_create(&_src, src, AML_LAYOUT_ORDER_FORTRAN,
	                              size, 1, &dims, &stride, &pitch);
	if (err != AML_SUCCESS)
		return err;
	err = aml_layout_dense_create(&_dst, dst, AML_LAYOUT_ORDER_FORTRAN,
	                              size, 1, &dims, &stride, &pitch);
	if (err != AML_SUCCESS)
		goto out_with_src;

	// Submit request
	err = aml_dma_copy_custom(dma, _dst, _src, dma_op, dma_op_arg);

	aml_layout_destroy(&_dst);
out_with_src:
	aml_layout_destroy(&_src);
	return err;
}

ssize_t aml_mapper_mapped_size(struct aml_mapper *mapper,
                               void *ptr,
                               size_t num,
                               struct aml_dma *dma,
                               aml_dma_operator dma_op,
                               void *dma_op_arg)
{
	int err;
	size_t tot = 0;
	char _ptr[mapper->size], *p;
	void **src;

	// Get a temporary host copy of ptr.
	if (mapper->flags & AML_MAPPER_FLAG_SHALLOW || dma == NULL) {
		p = ptr;
	} else {
		err = aml_mapper_memcpy(ptr, _ptr, sizeof(_ptr), dma, dma_op,
		                        dma_op_arg);
		if (err != AML_SUCCESS)
			return (ssize_t)err;
		p = _ptr;
	}

	if (!(mapper->flags & AML_MAPPER_FLAG_SHALLOW))
		tot += mapper->size * num;

	// Add size of each field of `ptr`.
	for (size_t i = 0; i < mapper->n_fields; i++) {
		// If flag AML_MAPPER_FLAG_SPLIT is set then do not
		// count size.
		if (mapper->fields[i]->flags & AML_MAPPER_FLAG_SPLIT)
			continue;
		size_t n = get_num_elements(mapper, i, ptr);
		if (n == 0)
			continue;
		for (size_t j = 0; j < num; j++) {
			src = PTR_OFF(p, +,
			              mapper->size * j + mapper->offsets[i]);
			if (*src == NULL)
				continue;
			tot += aml_mapper_mapped_size(mapper->fields[i], *src,
			                              n, dma, dma_op,
			                              dma_op_arg);
		}
	}
	return tot;
}

static ssize_t mapper_mmap_recursive(struct aml_mapper *mapper,
                                     void *src,
                                     void *dst,
                                     size_t num,
                                     void *ptr,
                                     struct aml_area *area,
                                     struct aml_area_mmap_options *area_opts,
                                     struct aml_dma *dma,
                                     aml_dma_operator dma_op,
                                     void *dma_op_arg)
{
	ssize_t err, off = 0;
	const size_t size = mapper->size * num;
	// Keep track of all indirections for final copy.
	void *indirections[mapper->n_fields * num];
	memset(indirections, 0, sizeof(indirections));

	// For each field.
	for (size_t j = 0; j < mapper->n_fields; j++) {
		// Number of elements for each field.
		size_t n = get_num_elements(mapper, j, src);
		if (n == 0)
			continue;

		// For each array element.
		for (size_t i = 0; i < num; i++) {
			// src field pointer value.
			void *src_ptr = *(void **)PTR_OFF(
			        src, +, mapper->offsets[j] + i * mapper->size);
			if (src_ptr == NULL)
				continue;
			// If split flag is set then we need to allocate new
			// space.
			if (mapper->fields[j]->flags & AML_MAPPER_FLAG_SPLIT) {
				err = aml_mapper_mmap(
				        mapper->fields[j], src_ptr,
				        &indirections[j * num + i], n, area,
				        area_opts, dma, dma_op, dma_op_arg);
				if (err < 0)
					goto err_mmap;
			}
			// Else we recurse on field.
			else {
				err = mapper_mmap_recursive(
				        mapper->fields[j], src_ptr,
				        PTR_OFF(ptr, +, off), n,
				        PTR_OFF(ptr, +,
				                off + n * mapper->fields[j]
				                                        ->size),
				        area, area_opts, dma, dma_op,
				        dma_op_arg);
				if (err < 0)
					goto err_mmap;
				indirections[j * num + i] =
				        PTR_OFF(ptr, +, off);
				off += err;
			}
		}
	}

	// Recursion on child fields is over.
	// Now we can copy pointers and struct content.
	if (mapper->flags & AML_MAPPER_FLAG_SHALLOW) {
		memcpy(dst, src, size);
		for (size_t i = 0; i < num; i++)
			for (size_t j = 0; j < mapper->n_fields; j++)
				*(void **)PTR_OFF(dst, +,
				                  mapper->size * i +
				                          mapper->offsets[j]) =
				        indirections[j * num + i];
	} else {
		char *local_copy = malloc(size);
		if (local_copy == NULL) {
			err = -AML_ENOMEM;
			goto err_mmap;
		}
		// Copy struct
		memcpy(local_copy, src, size);
		// Copy indirections
		for (size_t i = 0; i < num; i++)
			for (size_t j = 0; j < mapper->n_fields; j++)
				*(void **)PTR_OFF(local_copy, +,
				                  mapper->size * i +
				                          mapper->offsets[j]) =
				        indirections[j * num + i];
		// Move buffer to device.
		err = aml_mapper_memcpy(local_copy, dst, size, dma, dma_op,
		                        dma_op_arg);
		free(local_copy);
		if (err != AML_SUCCESS)
			goto err_mmap;
	}

	// Everything went fine! Return advances in pointer.
	return size + off;

err_mmap:
	for (size_t j = 0; j < mapper->n_fields; j++)
		if (mapper->fields[j]->flags & AML_MAPPER_FLAG_SPLIT) {
			size_t n = get_num_elements(mapper, j, src);
			for (size_t i = 0; i < num; i++)
				if (indirections[j * num + i] != NULL) {
					void *src_ptr = *(void **)PTR_OFF(
					        src, +,
					        mapper->offsets[j] +
					                i * mapper->size);
					aml_mapper_munmap(
					        mapper->fields[j],
					        indirections[j * num + i], n,
					        src_ptr, area, dma, dma_op,
					        dma_op_arg);
				}
		}
	return err;
}

int aml_mapper_mmap(struct aml_mapper *mapper,
                    void *src,
                    void *dst,
                    size_t num,
                    struct aml_area *area,
                    struct aml_area_mmap_options *area_opts,
                    struct aml_dma *dma,
                    aml_dma_operator dma_op,
                    void *dma_op_arg)
{
	if (src == NULL || dst == NULL || mapper == NULL || area == NULL ||
	    dma == NULL)
		return -AML_EINVAL;

	ssize_t err;
	void *out;

	// Alloc pointer
	ssize_t size =
	        aml_mapper_mapped_size(mapper, src, num, NULL, NULL, NULL);
	if (size < 0)
		return size;
	out = aml_area_mmap(area, size, area_opts);
	if (out == NULL)
		return -AML_ENOMEM;

	// Map recursively in allocated space.
	if (mapper->flags & AML_MAPPER_FLAG_SHALLOW)
		err = mapper_mmap_recursive(mapper, src, dst, num, out, area,
		                            area_opts, dma, dma_op, dma_op_arg);
	else
		err = mapper_mmap_recursive(mapper, src, out, num,
		                            PTR_OFF(out, +, mapper->size * num),
		                            area, area_opts, dma, dma_op,
		                            dma_op_arg);
	if (err < 0) {
		aml_area_munmap(area, &out, size);
		return (int)err;
	}

	if (!(mapper->flags & AML_MAPPER_FLAG_SHALLOW))
		*(void **)dst = out;
	return AML_SUCCESS;
}

int aml_mapper_copy(struct aml_mapper *mapper,
                    void *src,
                    void *dst,
                    size_t num,
                    struct aml_dma *dma,
                    aml_dma_operator dma_op,
                    void *dma_op_arg)
{
	int err;
	void **dst_field;
	void *src_field; // Cannot dereference device pointer

	size_t n[mapper->n_fields];
	for (size_t j = 0; j < mapper->n_fields; j++)
		n[j] = get_num_elements(mapper, j, dst);

	if (mapper->flags & AML_MAPPER_FLAG_COPY) {
		// Save all indirections.
		void *ptr[mapper->n_fields * num];
		for (size_t i = 0; i < num; i++)
			for (size_t j = 0; j < mapper->n_fields; j++)
				ptr[i * mapper->n_fields + j] =
				        *(void **)PTR_OFF(
				                dst, +,
				                i * mapper->size +
				                        mapper->offsets[j]);

		// Do one big copy.
		if (mapper->flags & AML_MAPPER_FLAG_SHALLOW) {
			memcpy(dst, src, mapper->size * num);
		} else {
			err = aml_mapper_memcpy(src, dst, mapper->size * num,
			                        dma, dma_op, dma_op_arg);
			if (err != AML_SUCCESS)
				return err;
		}

		// Restore indirections
		for (size_t i = 0; i < num; i++)
			for (size_t j = 0; j < mapper->n_fields; j++) {
				dst_field = PTR_OFF(dst, +,
				                    i * mapper->size +
				                            mapper->offsets[j]);
				*dst_field = ptr[i * mapper->n_fields + j];
			}
	}

	// Recurse for each pointer
	for (size_t j = 0; j < mapper->n_fields; j++) {
		if (n[j] == 0)
			continue;
		for (size_t i = 0; i < num; i++) {

			dst_field = PTR_OFF(
			        dst, +, i * mapper->size + mapper->offsets[j]);
			src_field = PTR_OFF(
			        src, +, i * mapper->size + mapper->offsets[j]);
			// Get device pointer at position src_field and copy it
			// in the same variable.
			if (mapper->flags & AML_MAPPER_FLAG_SHALLOW)
				src_field = *(void **)src_field;
			else {
				err = aml_mapper_memcpy(src_field, &src_field,
				                        sizeof(src_field), dma,
				                        dma_op, dma_op_arg);
				if (err != AML_SUCCESS)
					return err;
			}

			if (src_field == NULL)
				continue;

			err = aml_mapper_copy(mapper->fields[j], src_field,
			                      *dst_field, n[j], dma, dma_op,
			                      dma_op_arg);
			if (err != AML_SUCCESS)
				return err;
		}
	}

	return AML_SUCCESS;
}

ssize_t aml_mapper_munmap(struct aml_mapper *mapper,
                          void *ptr,
                          size_t num,
                          void *src,
                          struct aml_area *area,
                          struct aml_dma *dma,
                          aml_dma_operator dma_op,
                          void *dma_op_arg)
{
	size_t s = 0;
	void *field_ptr;
	ssize_t err;

	// Recurse to unmap every children first.
	for (size_t i = 0; i < mapper->n_fields; i++) {
		size_t n = get_num_elements(mapper, i, src);
		for (size_t j = 0; j < num; j++) {
			// Get pointer to child field.
			err = aml_mapper_memcpy(
			        PTR_OFF(ptr, +,
			                mapper->size * j + mapper->offsets[i]),
			        &field_ptr, sizeof(field_ptr), dma, dma_op,
			        dma_op_arg);
			if (err < 0)
				return err;
			if (field_ptr == NULL)
				continue;
			err = aml_mapper_munmap(
			        mapper->fields[i], field_ptr, n,
			        *(void **)PTR_OFF(src, +,
			                          mapper->size * j +
			                                  mapper->offsets[i]),
			        area, dma, dma_op, dma_op_arg);
			if (err < 0)
				return err;
			s += err;
		}
	}

	// Unmap this one
	// If this structure was split, then break the recursion
	// after unmapping data.
	if (mapper->flags & AML_MAPPER_FLAG_SPLIT) {
		aml_area_munmap(area, &ptr, s + mapper->size * num);
		return 0;
	} else if (mapper->flags & AML_MAPPER_FLAG_SHALLOW &&
	           mapper->n_fields > 0) {
		aml_area_munmap(area, PTR_OFF(ptr, +, mapper->offsets[0]), s);
		return 0;
	}
	// If this structure was not split, then return size of it
	// non-split children.
	return mapper->size + s;
}
