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

#include "aml/higher/allocator/utils.h"
#include "aml/utils/utarray.h"

#define PTR_OFF(ptr, sign, off) (void *)((intptr_t)(ptr)sign(intptr_t)(off))
#define PTR_OP(a, sign, b) ((intptr_t)(a)sign(intptr_t)(b))
#define ALIGNED_PTR(ptr, s) PTR_OFF(ptr, +, s - ((intptr_t)ptr % s))

int aml_memory_chunk_comp(const void *lhs, const void *rhs)
{
	struct aml_memory_chunk *l = (struct aml_memory_chunk *)lhs;
	struct aml_memory_chunk *r = (struct aml_memory_chunk *)rhs;

	if (l->ptr == r->ptr)
		return 0;
	else if (l->ptr < r->ptr)
		return -1;
	else
		return 1;
}

UT_icd aml_memory_chunk_icd = {
        .sz = sizeof(struct aml_memory_chunk),
        .init = NULL,
        .copy = NULL,
        .dtor = NULL,
};

int aml_memory_chunk_match(struct aml_memory_chunk a, struct aml_memory_chunk b)
{
	return (size_t)a.ptr + a.size == (size_t)b.ptr ||
	       (size_t)b.ptr + b.size == (size_t)a.ptr;
}

int aml_memory_chunk_overlap(struct aml_memory_chunk a,
                             struct aml_memory_chunk b)
{
	if ((uintptr_t)a.ptr <= (uintptr_t)b.ptr &&
	    (uintptr_t)b.ptr - (uintptr_t)a.ptr < a.size)
		return 1;
	if ((uintptr_t)b.ptr <= (uintptr_t)a.ptr &&
	    (uintptr_t)a.ptr - (uintptr_t)b.ptr < b.size)
		return 1;
	return 0;
}

int aml_memory_chunk_contains(struct aml_memory_chunk super,
                              struct aml_memory_chunk sub)
{
	if ((intptr_t)sub.ptr < (intptr_t)super.ptr)
		return 0;
	if ((intptr_t)sub.ptr + sub.size > (intptr_t)super.ptr + super.size)
		return 0;
	return 1;
}

int aml_memory_pool_create(struct aml_memory_pool **out,
                           void *ptr,
                           const size_t size)
{
	struct aml_memory_pool *pool;
	UT_array *chunks;

	if (out == NULL || size == 0)
		return -AML_EINVAL;

	// Allocate pool.
	pool = malloc(sizeof(*pool));
	if (pool == NULL)
		return -AML_ENOMEM;

	// Allocate queue with chunks
	utarray_new(chunks, &aml_memory_chunk_icd);
	pool->chunks = (void *)chunks;

	// Set memory of the pool.
	pool->memory.size = size;
	pool->memory.ptr = ptr;

	// Push initial chunk in queue.
	// Should not fail because vector has available space when it is
	// created.
	utarray_push_back(chunks, &pool->memory);

	*out = pool;
	return AML_SUCCESS;
}

int aml_memory_pool_destroy(struct aml_memory_pool **out)
{
	if (out == NULL || *out == NULL)
		return -AML_EINVAL;

	UT_array *chunks = (UT_array *)(*out)->chunks;
	utarray_free(chunks);
	free(*out);
	*out = NULL;
	return AML_SUCCESS;
}

int aml_memory_pool_push(struct aml_memory_pool *pool,
                         void *ptr,
                         const size_t size)
{
	if (pool == NULL || size == 0)
		return -AML_EINVAL;

	UT_array *chunks = (UT_array *)pool->chunks;
	struct aml_memory_chunk new_chunk = {.ptr = ptr, .size = size};
	struct aml_memory_chunk *chunk;

	// Out of bound
	if (!aml_memory_chunk_contains(pool->memory, new_chunk))
		return -AML_EDOM;

	// Iterate through the pool to find a matching chunk.
	for (size_t i = 0; i < utarray_len(chunks); i++) {
		chunk = utarray_eltptr(chunks, i);
		// If chunks overlap it is an unexpected bug.
		if (aml_memory_chunk_overlap(*chunk, new_chunk))
			return -AML_EINVAL;

		// If chunks match, we concatenate them.
		if (aml_memory_chunk_match(*chunk, new_chunk)) {
			if ((intptr_t)new_chunk.ptr < (intptr_t)chunk->ptr)
				chunk->ptr = new_chunk.ptr;
			chunk->size += new_chunk.size;
			return AML_SUCCESS;
		}
	}

	// If no matching chunk was found we push this chunk in the vector.
	utarray_push_back(chunks, &new_chunk);
	return AML_SUCCESS;
}

int aml_memory_pool_pop(struct aml_memory_pool *pool,
                        void **out,
                        const size_t size)
{
	struct aml_memory_chunk *chunk;
	UT_array *chunks = (UT_array *)pool->chunks;
	size_t len = utarray_len(chunks);

	// Empty pool.
	if (len == 0)
		return -AML_ENOMEM;

	// If last chunk is the exact size, we pop it.
	chunk = utarray_eltptr(chunks, len - 1);
	if (chunk->size == size) {
		*out = chunk->ptr;
		utarray_pop_back(chunks);
		return AML_SUCCESS;
	}

	// Walk the vector to find a chunk large enough
	for (ssize_t i = utarray_len(chunks) - 1; i >= 0; i--) {
		chunk = utarray_eltptr(chunks, i);
		if (chunk->size > size)
			goto chunk;
		if (chunk->size == size)
			goto take;
	}

	// Try to merge matching chunks.
	utarray_sort(chunks, aml_memory_chunk_comp);
	for (size_t i = 1; i < utarray_len(chunks); i++) {
		struct aml_memory_chunk *prev = utarray_eltptr(chunks, i - 1);
		struct aml_memory_chunk *curr = utarray_eltptr(chunks, i);

		if (aml_memory_chunk_match(*prev, *curr)) {
			prev->size += curr->size;
			utarray_erase(chunks, i, 1);
			i--;
		}
	}

	// Walk the vector to find a chunk large enough
	for (ssize_t i = utarray_len(chunks) - 1; i >= 0; i--) {
		chunk = utarray_eltptr(chunks, i);
		if (chunk->size > size)
			goto chunk;
		if (chunk->size == size)
			goto take;
	}

	// No chunk available.
	return -AML_ENOMEM;

chunk:
	chunk->size -= size;
	*out = PTR_OFF(chunk->ptr, +, chunk->size);
	return AML_SUCCESS;
take:
	*out = chunk->ptr;
	utarray_erase(chunks, utarray_eltidx(chunks, chunk), 1);
	return AML_SUCCESS;
}
