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
static int
aml_mapper_copy(void *src, void *dst, size_t size, struct aml_mapper_args *args)
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
	err = aml_dma_copy_custom(args->dma, _dst, _src, args->dma_op,
	                          args->dma_op_arg);
	if (err != AML_SUCCESS)
		goto out_with_dst;

out_with_dst:
	aml_layout_destroy(&_dst);
out_with_src:
	aml_layout_destroy(&_src);
	return err;
}

/**
 * Helper function reusable in the library to get the memory
 * mapped size of a mapper.
 * Child fields with flag AML_MAPPER_FLAG_SPLIT set are not counted in this
 * size because they are allcoated separately.
 * @arg mapper: The mapper representing allocated `ptr`.
 * @arg args: If ptr is not a host pointer, args must contain a set dma
 * arguments to copy data from device to host.
 * @arg ptr: The pointer for which to compute size according to its `mapper`
 * description.
 * @return A size on success, an AML error code returned by dma calls on error.
 */
ssize_t aml_mapper_mapped_size(struct aml_mapper *mapper,
                               struct aml_mapper_args *args,
                               void *ptr,
                               size_t num)
{
	int err;
	size_t tot = 0;
	char _ptr[mapper->size], *p;
	void **src;
	size_t n;

	// Get a temporary host copy of ptr.
	if (args != NULL) {
		err = aml_mapper_copy(ptr, _ptr, sizeof(_ptr), args);
		if (err != AML_SUCCESS)
			return (ssize_t)err;
		p = _ptr;
	} else {
		p = ptr;
	}

	if (!(mapper->flags & AML_MAPPER_FLAG_SHALLOW))
		tot += mapper->size;

	// Add size of each field of `ptr`.
	for (size_t i = 0; i < mapper->n_fields; i++) {
		// If flag AML_MAPPER_FLAG_SPLIT is set then do not
		// count size.
		if (mapper->fields[i]->flags & AML_MAPPER_FLAG_SPLIT)
			continue;
		n = get_num_elements(mapper, i, ptr);
		for (size_t j = 0; j < num; j++) {
			src = PTR_OFF(p, +, mapper->size * j + mapper->offsets[i]);
			tot += aml_mapper_mapped_size(mapper->fields[i], args, *src, n);
		}
	}
	return tot;
}

// Helper macro to navigate allocated host pointer
#define GET_PTR(ptr, mapper, n, field)                                         \
	*(void **)PTR_OFF(ptr, +, (n * mapper->size + mapper->offsets[field]))

// Helper macro to cleanup in allocated host.
#define CLEAN_PTR(ptr, mapper, args, n, field)                                 \
	if (mapper->fields[field]->flags & AML_MAPPER_FLAG_SPLIT) {            \
		aml_mapper_munmap(mapper->fields[field], args,                 \
		                  GET_PTR(ptr, mapper, n, field));             \
	}

/**
 * Map `src` pointer containing `num` times the structure described by `mapper`
 * into mapped `dst` pointer and copying with dma engine in `args`.
 * `src` must be a host pointer because it is dereferenced.
 * Mapped `dst` pointer is assumed to contain enough space to contiguously
 * copy `num` times structure described by `mapper`.
 */
static ssize_t aml_mapper_mmap_mapped(struct aml_mapper *mapper,
                                      struct aml_mapper_args *args,
                                      void *src,
                                      void *dst,
                                      size_t num)
{
	ssize_t inc, off = 0;
	const size_t size = mapper->size * num;

	// Pointer to available space.
	void *ptr = PTR_OFF(dst, +, size);
	// Pointer to src/dst fields.
	void *src_field, *dst_field;
	// Number of elements for each field.
	size_t n[mapper->n_fields];
	for (size_t j = 0; j < mapper->n_fields; j++)
		n[j] = get_num_elements(mapper, j, src);

	// For each array element.
	for (size_t i = 0; i < num; i++) {
		// For each element field.
		for (size_t j = 0; j < mapper->n_fields; j++) {
			// Get offset of field.
			inc = i * mapper->size + mapper->offsets[j];
			// Get pointer to src/dst field structs.
			src_field = *(void **)PTR_OFF(src, +, inc);
			dst_field = PTR_OFF(dst, +, inc); // cannot deref device
			                                  // pointer.

			// If split flag is set then we need to allocate new
			// space. It also applies if the structure is top level
			// shallow copy.
			if (mapper->fields[j]->flags & AML_MAPPER_FLAG_SPLIT ||
			    mapper->flags & AML_MAPPER_FLAG_SHALLOW) {
				off = aml_mapper_mmap(mapper->fields[j], args,
				                      src_field, dst_field,
				                      n[j]);
				if (off < 0)
					goto err;
				off = 0; // Do not increment in the mapped
				         // space.
			}
			// Else we recurse on field.
			else {
				// If flag AML_MAPPER_FLAG_COPY is set, then
				// copy the structure into available space.
				if (mapper->flags & AML_MAPPER_FLAG_COPY) {
					off = aml_mapper_copy(src_field, ptr,
					                      size, args);
					if (off < 0)
						goto err;
				}
				// Copy pointer to available space value into
				// the right field.
				off = aml_mapper_copy(&ptr, dst_field,
				                      sizeof(ptr), args);
				if (off != AML_SUCCESS)
					goto err;
				// Recurse mapping of field src_field into free
				// space (ptr).
				off = aml_mapper_mmap_mapped(mapper->fields[j],
				                             args, src_field,
				                             ptr, n[j]);
				if (off < 0)
					goto err;
			}
			// Increment ptr by the advances made by recursive
			// calls
			ptr = PTR_OFF(ptr, +, off);

			continue;
			// Handle errors and cleanup
		err:
			while (j--)
				CLEAN_PTR(dst, mapper, args, i, j);
			while (i--)
				for (size_t j = 0; j < mapper->n_fields; j++) {
					CLEAN_PTR(dst, mapper, args, i, j);
				}
			return off;
		}
	}

	return (ssize_t)ptr - (ssize_t)dst;
}

ssize_t aml_mapper_mmap(struct aml_mapper *mapper,
                        struct aml_mapper_args *args,
                        void *src,
                        void *dst,
                        size_t num)
{
	if (src == NULL || dst == NULL || mapper == NULL || args == NULL)
		return -AML_EINVAL;

	size_t size = aml_mapper_mapped_size(mapper, NULL, src, num);
	void *out;
	ssize_t err;

	// dst is an host pointer. Then no allocation is needed.
	if (mapper->flags & AML_MAPPER_FLAG_SHALLOW) {
		// If flag AML_MAPPER_FLAG_COPY is set, then copy the full
		// pointer content.
		if (mapper->flags & AML_MAPPER_FLAG_COPY)
			memcpy(dst, src, mapper->size * num);
		out = dst;
	} else {
		// Alloc pointer
		out = aml_area_mmap(args->area, size, args->area_opts);
		if (out == NULL)
			return -AML_ENOMEM;
		// Copy pointer value into destination pointer.
		err = aml_mapper_copy(&out, (void **)dst, sizeof(out), args);
		if (err < 0) {
			aml_area_munmap(args->area, &out, size);
			return err;
		}
		// If flag AML_MAPPER_FLAG_COPY is set, then copy the full
		// pointer content.
		if (mapper->flags & AML_MAPPER_FLAG_COPY) {
			err = aml_mapper_copy(src, out, mapper->size * num, args);
			if (err < 0) {
				aml_area_munmap(args->area, &out, size);
				return err;
			}
		}
	}

	// Map recursively in allocated space.
	err = aml_mapper_mmap_mapped(mapper, args, src, out, num);
	if (err < 0) {
		aml_area_munmap(args->area, &out, size);
		return err;
	}

	return size;
}

int aml_mapper_copy_back(struct aml_mapper *mapper,
                         struct aml_mapper_args *args,
                         void *src,
                         void *dst,
                         size_t num)
{
	int err;
	void **dst_field;
	void *src_field; // Cannot dereference device pointer

	size_t n[mapper->n_fields];
	for (size_t j = 0; j < mapper->n_fields; j++)
		n[j] = get_num_elements(mapper, j, src);

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
			err = aml_mapper_copy(src, dst, mapper->size * num,
			                      args);
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
	for (size_t i = 0; i < num; i++)
		for (size_t j = 0; j < mapper->n_fields; j++) {
			dst_field = PTR_OFF(
			        dst, +, i * mapper->size + mapper->offsets[j]);
			src_field = PTR_OFF(
			        src, +, i * mapper->size + mapper->offsets[j]);
			// Get device pointer at position src_field and copy it
			// in the same variable.
			err = aml_mapper_copy(src_field, &src_field,
			                      sizeof(src_field), args);
			if (err != AML_SUCCESS)
				return err;

			err = aml_mapper_copy_back(mapper->fields[j], args,
			                           src_field, *dst_field, n[j]);
			if (err != AML_SUCCESS)
				return err;
		}

	return AML_SUCCESS;
}

size_t aml_mapper_munmap(struct aml_mapper *mapper,
                         struct aml_mapper_args *args,
                         void *ptr)
{
	size_t s = 0;
	void *field_ptr;
	for (size_t i = 0; i < mapper->n_fields; i++) {
		// Get pointer to child field.
		aml_mapper_copy(PTR_OFF(ptr, +, mapper->offsets[i]), &field_ptr,
		                sizeof(field_ptr), args);
		// Recursively unmap
		s += aml_mapper_munmap(mapper->fields[i], args, field_ptr);
	}
	// If this structure was split, then break the recursion.
	// Unmap data and return 0 so that parent calls do not
	// account for this mapper size.
	if (mapper->flags & AML_MAPPER_FLAG_SPLIT) {
		aml_area_munmap(args->area, &ptr, s);
		return 0;
	}
	// If this structure was not split, then return size of it
	// non-split children.
	return mapper->size + s;
}
