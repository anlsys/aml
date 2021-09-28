/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include "aml.h"

#include "aml/higher/allocator.h"
#include "aml/higher/allocator/sized.h"
#include "aml/higher/allocator/utils.h"
#include "aml/utils/inner-malloc.h"
#include "aml/utils/utarray.h"

#define PTR_OFF(ptr, sign, off) (void *)((intptr_t)(ptr)sign(intptr_t)(off))
#define ALIGNED_PTR(ptr, s) PTR_OFF(ptr, +, s - ((intptr_t)ptr % s))

UT_icd aml_memory_pool_icd = {
        .sz = sizeof(struct aml_memory_pool *),
        .init = NULL,
        .copy = NULL,
        .dtor = NULL,
};

/** Add a new memory pool to allocator. */
static int aml_allocator_sized_extend(struct aml_allocator_sized *data,
                                      size_t size,
                                      struct aml_memory_pool **out)
{
	int err;
	void *ptr;
	struct aml_memory_pool *pool;
	UT_array *array = (UT_array *)data->pools;
	const size_t len = utarray_len(array);
	struct aml_memory_pool **pools =
	        (struct aml_memory_pool **)utarray_eltptr(array, 0);
	size_t total_size = 0;

	// Get total allocated size.
	for (size_t i = 0; i < len; i++)
		total_size += pools[i]->memory.size;
	if (total_size <= size)
		total_size = size;

	// Try to allocate double of total size to avoid calling this function
	// too frequently.
	size_t alloc_size = total_size;
	while (alloc_size >= size &&
	       (ptr = aml_area_mmap(data->area, alloc_size, data->opts)) ==
	               NULL)
		alloc_size /= 2;
	if (ptr == NULL)
		return aml_errno;

	// Create memory pool.
	err = aml_memory_pool_create(&pool, ptr, alloc_size);
	if (err != AML_SUCCESS)
		goto err_with_ptr;

	// Push new pool into the vector of pools.
	utarray_push_back(array, &pool);

	if (out != NULL)
		*out = pool;
	return AML_SUCCESS;

err_with_ptr:
	aml_area_munmap(data->area, ptr, alloc_size);
	return err;
}

int aml_allocator_sized_create(struct aml_allocator **allocator,
                               size_t size,
                               struct aml_area *area,
                               struct aml_area_mmap_options *opts)
{
	int err = -AML_ENOMEM;
	struct aml_allocator *alloc;
	struct aml_allocator_sized *data;
	UT_array *pools;

	// Allocate high level structure and metadata.
	alloc = AML_INNER_MALLOC(struct aml_allocator,
	                         struct aml_allocator_sized);
	if (alloc == NULL)
		return -AML_ENOMEM;
	alloc->data = AML_INNER_MALLOC_GET_FIELD(alloc, 2, struct aml_allocator,
	                                         struct aml_allocator_sized);
	data = (struct aml_allocator_sized *)alloc->data;
	alloc->ops = &aml_allocator_sized;

	// Fill metadata
	data->chunk_size = size;
	data->area = area;
	data->opts = opts;

	// Allocate space for pools.
	utarray_new(pools, &aml_memory_pool_icd);
	data->pools = (void *)pools;

	// Create first pool.
	const size_t pool_size = 1UL << 20; // 1MiB
	err = aml_allocator_sized_extend(data, pool_size, NULL);
	if (err != AML_SUCCESS)
		goto err_with_vec;

	*allocator = alloc;
	return AML_SUCCESS;

err_with_vec:
	utarray_free(pools);
	free(alloc);
	return err;
}

int aml_allocator_sized_destroy(struct aml_allocator **allocator)
{
	if (allocator == NULL || *allocator == NULL ||
	    (*allocator)->data == NULL)
		return -AML_EINVAL;

	int err;
	struct aml_allocator_sized *data =
	        (struct aml_allocator_sized *)(*allocator)->data;
	UT_array *pools = (UT_array *)data->pools;
	unsigned n = utarray_len(pools);

	for (ssize_t i = n - 1; i >= 0; i--) {
		struct aml_memory_pool *pool =
		        *(struct aml_memory_pool **)utarray_eltptr(pools, i);
		aml_area_munmap(data->area, pool->memory.ptr,
		                pool->memory.size);
		err = aml_memory_pool_destroy(&pool);
		if (err != AML_SUCCESS)
			return err;
		utarray_pop_back(pools);
	}

	utarray_free(pools);
	free(*allocator);
	*allocator = NULL;

	return AML_SUCCESS;
}

void *aml_allocator_sized_alloc(struct aml_allocator_data *data, size_t size)
{
	int err;
	void *ptr;
	struct aml_allocator_sized *alloc = (struct aml_allocator_sized *)data;
	struct aml_memory_pool *pool;
	UT_array *pools = (UT_array *)alloc->pools;

	// Check input
	if (size > alloc->chunk_size) {
		aml_errno = -AML_EINVAL;
		return NULL;
	}
	size = alloc->chunk_size;

	// Try to pop a chunk from a pool.
	for (size_t i = 0; i < utarray_len(pools); i++) {
		pool = *(struct aml_memory_pool **)utarray_eltptr(pools, i);
		err = aml_memory_pool_pop(pool, &ptr, size);
		if (err == AML_SUCCESS)
			return ptr;
		if (err != -AML_ENOMEM) {
			aml_errno = err;
			return NULL;
		}
	}

	// All pools are empty. Let's extend pool list with a new pool.
	err = aml_allocator_sized_extend(alloc, size, &pool);
	if (err != AML_SUCCESS) {
		aml_errno = err;
		return NULL;
	}

	// Get a chunk from the new pool.
	err = aml_memory_pool_pop(pool, &ptr, size);
	if (err == AML_SUCCESS)
		return ptr;
	if (err != -AML_ENOMEM) {
		aml_errno = err;
		return NULL;
	}

	return ptr;
}

int aml_allocator_sized_free(struct aml_allocator_data *data, void *ptr)
{
	struct aml_allocator_sized *alloc = (struct aml_allocator_sized *)data;
	struct aml_memory_chunk chunk = {.ptr = ptr, .size = alloc->chunk_size};
	struct aml_memory_pool *pool;
	UT_array *pools = (UT_array *)alloc->pools;

	for (size_t i = 0; i < utarray_len(pools); i++) {
		pool = *(struct aml_memory_pool **)utarray_eltptr(pools, i);
		if (aml_memory_chunk_contains(pool->memory, chunk)) {
			return aml_memory_pool_push(pool, ptr,
			                            alloc->chunk_size);
		}
	}

	// This pointer is not from this allocator.
	return -AML_EINVAL;
}

struct aml_allocator_ops aml_allocator_sized = {
        .alloc = aml_allocator_sized_alloc,
        .aligned_alloc = NULL,
        .free = aml_allocator_sized_free,
};
