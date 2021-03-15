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

// Copy src into dst.
// Append the resulting request to a request queue.
static int aml_mapper_copy(void *src,
                           void *dst,
                           size_t size,
                           struct aml_dma *dma,
                           aml_dma_operator op,
                           void *op_arg)
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
	err = aml_dma_copy_custom(dma, _dst, _src, op, op_arg);
	if (err != AML_SUCCESS)
		goto out_with_dst;

out_with_dst:
	aml_layout_destroy(&_dst);
out_with_src:
	aml_layout_destroy(&_src);
	return err;
}

ssize_t aml_mapper_size(struct aml_mapper *mapper,
                        void *ptr,
                        struct aml_dma *dma,
                        aml_dma_operator op,
                        void *op_arg)
{
	int err;
	size_t tot = mapper->size;
	char _ptr[mapper->size], *p;

	if (dma != NULL) {
		err = aml_mapper_copy(ptr, _ptr, sizeof(_ptr), dma, op, op_arg);
		if (err != AML_SUCCESS)
			return (ssize_t)err;
		p = _ptr;
	} else {
		p = ptr;
	}

	for (size_t i = 0; i < mapper->n_fields; i++) {
		for (size_t j = 0; j < get_num_elements(mapper, i, p); j++) {
			tot += aml_mapper_size(
			        mapper->fields[i],
			        PTR_OFF(*(void **)PTR_OFF(p, +,
			                                  mapper->offsets[i]),
			                +, j * mapper->fields[i]->size),
			        dma, op, op_arg);
		}
	}
	return tot;
}

static ssize_t aml_mapper_map_field(struct aml_mapper *mapper,
                                    size_t num,
                                    void *src,
                                    void *dst,
                                    struct aml_dma *dma,
                                    aml_dma_operator op,
                                    void *op_arg)
{
	int err;
	ssize_t inc, off = 0;
	// Pointer to free space.
	void *ptr = PTR_OFF(dst, +, mapper->size * num), *_src;

	// Copy the full pointer content.
	if (mapper->flags & AML_MAPPER_FLAG_COPY) {
		err = aml_mapper_copy(src, dst, mapper->size * num, dma, op,
		                      op_arg);
		if (err != AML_SUCCESS)
			return err;
	}

	// For each array element.
	for (size_t i = 0; i < num; i++) {
		// For each element field.
		for (size_t j = 0; j < mapper->n_fields; j++) {
			// offset of field.
			inc = i * mapper->size + mapper->offsets[j];
			// src field content.
			_src = *(void **)PTR_OFF(src, +, inc);

			// Copy pointer to free space (dst + off) into dst field
			// pointer (dst + inc).
			err = aml_mapper_copy(&ptr, PTR_OFF(dst, +, inc),
			                      sizeof(ptr), dma, op, op_arg);
			if (err != AML_SUCCESS)
				return err;

			// Map src field (src + inc) into dst free space (dst +
			// off).
			off = aml_mapper_map_field(mapper->fields[j],
			                           get_num_elements(mapper, j,
			                                            src),
			                           _src, ptr, dma, op, op_arg);
			if (off <= AML_SUCCESS)
				return off;

			// Increment ptr by the advances made by recursive
			// calls
			ptr = PTR_OFF(ptr, +, off);
		}
	}

	return (ssize_t)ptr - (ssize_t)dst;
}

void *aml_mapper_mmap(struct aml_mapper *mapper,
                      void *ptr,
                      struct aml_area *area,
                      struct aml_area_mmap_options *opts,
                      struct aml_dma *dma,
                      aml_dma_operator op,
                      void *op_arg,
                      size_t *size)
{
	if (ptr == NULL || mapper == NULL || area == NULL || dma == NULL) {
		aml_errno = -AML_EINVAL;
		return NULL;
	}

	size_t _s, s;
	void *out;

	s = aml_mapper_size(mapper, ptr, NULL, NULL, NULL);
	out = aml_area_mmap(area, s, opts);
	if (out == NULL)
		return NULL;

	_s = aml_mapper_map_field(mapper, 1, ptr, out, dma, op, op_arg);
	if (_s != s) {
		aml_area_munmap(area, &out, s);
		aml_errno = _s;
		return NULL;
	}
	*size = s;

	return out;
}

int aml_mapper_shallow_mmap(struct aml_mapper *mapper,
                            void *src,
                            void *dst,
                            struct aml_area *area,
                            struct aml_area_mmap_options *opts,
                            struct aml_dma *dma,
                            aml_dma_operator op,
                            void *op_arg,
                            size_t *size)
{
	if (dst == NULL || src == NULL || mapper == NULL || area == NULL ||
	    dma == NULL) {
		return -AML_EINVAL;
	}

	size_t s = 0;
	ssize_t err;
	void *out;

	memcpy(dst, src, mapper->size);
	for (size_t i = 0; i < mapper->n_fields; i++) {
		if (mapper->fields[i]->n_fields == 0) {
			s += mapper->fields[i]->size *
			     mapper->num_elements[i](src);
		} else {
			for (size_t j = 0; j < mapper->num_elements[i](src);
			     j++)
				s += aml_mapper_size(
				        mapper->fields[i],
				        PTR_OFF(*(void **)PTR_OFF(
				                        src, +,
				                        mapper->offsets[i]),
				                +, j * mapper->fields[i]->size),
				        NULL, NULL, NULL);
		}
	}
	if (size != NULL)
		*size = s;

	if (s == 0)
		return AML_SUCCESS;

	out = aml_area_mmap(area, s, opts);
	if (out == NULL)
		return -AML_ENOMEM;

	void *dst_ptr = out;
	for (size_t i = 0; i < mapper->n_fields; i++) {
		void *src_ptr = *(void **)PTR_OFF(src, +, mapper->offsets[i]);
		// Advance dst_ptr
		if (i > 0) // Err is set to previous field mmap size.
			dst_ptr = PTR_OFF(dst_ptr, +, err);

		err = aml_mapper_map_field(mapper->fields[i],
		                           mapper->num_elements[i](src),
		                           src_ptr, dst_ptr, dma, op, op_arg);
		if (err < 0) {
			aml_area_munmap(area, &out, s);
			return err;
		}
		*(void **)PTR_OFF(dst, +, mapper->offsets[i]) = dst_ptr;
	}

	return AML_SUCCESS;
}

static int aml_mapper_copy_recursive(struct aml_mapper *mapper,
                                     size_t num,
                                     void *src,
                                     void *dst,
                                     struct aml_dma *dma,
                                     aml_dma_operator op,
                                     void *op_arg)
{
	int err;
	void *ptr[mapper->n_fields * num];

	// Save all indirections.
	for (size_t i = 0; i < num; i++)
		for (size_t j = 0; j < mapper->n_fields; j++)
			ptr[i * mapper->n_fields + j] = *(void **)PTR_OFF(
			        dst, +, i * mapper->size + mapper->offsets[j]);

	// Do one big copy.
	if (mapper->flags & AML_MAPPER_FLAG_COPY) {
		err = aml_mapper_copy(src, dst, mapper->size * num, dma, op,
		                      op_arg);
		if (err != AML_SUCCESS)
			return err;
	}

	// Restore indirections and Recurse for each pointer.
	for (size_t i = 0; i < num; i++)
		for (size_t j = 0; j < mapper->n_fields; j++) {
			void *_src = *(void **)PTR_OFF(
			        dst, +, i * mapper->size + mapper->offsets[j]);
			void *_dst = ptr[i * mapper->n_fields + j];
			*(void **)PTR_OFF(dst, +,
			                  i * mapper->size +
			                          mapper->offsets[j]) = _dst;
			err = aml_mapper_copy_recursive(
			        mapper->fields[j],
			        get_num_elements(mapper, j, dst), _src, _dst,
			        dma, op, op_arg);
			if (err != AML_SUCCESS)
				return err;
		}

	return AML_SUCCESS;
}

int aml_mapper_copy_back(struct aml_mapper *mapper,
                         void *src,
                         void *dst,
                         struct aml_dma *dma,
                         aml_dma_operator op,
                         void *op_arg)
{
	return aml_mapper_copy_recursive(mapper, 1, src, dst, dma, op, op_arg);
}

void aml_mapper_munmap(struct aml_mapper *mapper,
                       void *ptr,
                       struct aml_area *area,
                       struct aml_dma *dma,
                       aml_dma_operator op,
                       void *op_arg)
{
	aml_area_munmap(area, &ptr,
	                aml_mapper_size(mapper, ptr, dma, op, op_arg));
}

void aml_mapper_shallow_munmap(struct aml_mapper *mapper,
                               void *ptr,
                               struct aml_area *area,
                               struct aml_dma *dma,
                               aml_dma_operator op,
                               void *op_arg)
{
	size_t s = 0;
	void *data_ptr = *(void **)PTR_OFF(ptr, +, mapper->offsets[0]);
	for (size_t i = 0; i < mapper->n_fields; i++) {
		s += aml_mapper_size(mapper->fields[i],
		                     *(void **)PTR_OFF(ptr, +,
		                                       mapper->offsets[i]),
		                     dma, op, op_arg);
	}
	if (s > 0)
		aml_area_munmap(area, &data_ptr, s);
}
