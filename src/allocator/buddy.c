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
#include "aml/higher/allocator/buddy.h"

//------------------------------------------------------------------------------
// Buddy allocator implementation
//------------------------------------------------------------------------------

#define utarray_oom()                                                          \
	do {                                                                   \
		err = -AML_ENOMEM;                                             \
	} while (0)

#define HASH_NONFATAL_OOM 1
#define uthash_nonfatal_oom(obj)                                               \
	do {                                                                   \
		err = -AML_ENOMEM;                                             \
	} while (0)

#include "internal/utarray.h"
#include "internal/uthash.h"
#include "internal/utlist.h"

static int is_power_of_two(size_t size)
{
	if (size == 0)
		return 0;
	return (size & (size - 1)) == 0;
}

// Get the rank associated with a size.
// The greatest size gets rank 0.
// Rank increases when size decreases.
static size_t buddy_rank(size_t size)
{
	assert(size != 0);
	size_t i = 1;

	// Shift left until we meet a bit that is not 0.
	while (((size << i) >> i) == size)
		i++;

	return i - 1;
}

// The smallest chunk index in ranks array.
// The corresponding size is 2^6 = 64 bytes.
// The allocator won't accept allocations smaller than this.
#define _MAX_BUDDY_RANK ((sizeof(size_t) * 8) - 6)
const size_t MAX_BUDDY_RANK = _MAX_BUDDY_RANK;
const size_t MIN_SIZE = 1 + (~0UL >> _MAX_BUDDY_RANK);

struct buddy_chunk {
	void *ptr;
	size_t size;
	UT_hash_handle hh;
	// Utlist pointer to next chunk.
	struct buddy_chunk *next;
	// Utlist pointer to previous chunk.
	struct buddy_chunk *prev;
	// The adjascent chunk on the right.
	struct buddy_chunk *next_buddy;
	// The adjascent chunk on the left.
	struct buddy_chunk *prev_buddy;
	// Wheather this chunk is allocated or not.
	int is_allocated;
};

static struct buddy_chunk *buddy_chunk_create(void *ptr,
                                              size_t size,
                                              struct buddy_chunk *next_buddy,
                                              struct buddy_chunk *prev_buddy)
{
	struct buddy_chunk *chunk = malloc(sizeof(*chunk));
	if (chunk == NULL)
		return NULL;
	chunk->ptr = ptr;
	chunk->size = size;
	chunk->next = NULL;
	chunk->prev = NULL;
	chunk->next_buddy = next_buddy;
	if (next_buddy != NULL)
		next_buddy->prev_buddy = chunk;
	chunk->prev_buddy = prev_buddy;
	if (prev_buddy != NULL)
		prev_buddy->next_buddy = chunk;
	chunk->is_allocated = 0;
	return chunk;
}

int buddy_chunk_comp(const void *lhs, const void *rhs)
{
	struct buddy_chunk *l = (struct buddy_chunk *)lhs;
	struct buddy_chunk *r = (struct buddy_chunk *)rhs;

	if (l->ptr == r->ptr)
		return 0;
	else if (l->ptr < r->ptr)
		return -1;
	else
		return 1;
}

UT_icd ranks_icd = {
        .sz = sizeof(struct buddy_chunk *),
        .init = NULL,
        .copy = NULL,
        .dtor = NULL,
};

// Low-level allocator definition.
struct buddy_allocator {
	// Array of lists. For each index the corresponding list contains blocks
	// of size 2^(64-index).
	UT_array *ranks;
	// Hash table of allocation on flight.
	struct buddy_chunk *allocations;
};

int buddy_allocator_extend(struct buddy_allocator *b, void *ptr, size_t size)
{
	if (ptr == NULL || !is_power_of_two(size))
		return -AML_EINVAL;

	struct buddy_chunk **buddy, *chunk;
	size_t rank = buddy_rank(size);

	// Allocate the chunk to insert in the allocator.
	chunk = buddy_chunk_create(ptr, size, NULL, NULL);
	if (chunk == NULL)
		return -AML_ENOMEM;

	buddy = utarray_eltptr(b->ranks, rank);
	DL_APPEND(*buddy, chunk);

	return AML_SUCCESS;
}

int buddy_allocator_create(struct buddy_allocator **out)
{
	int err = AML_SUCCESS;
	struct buddy_allocator *b;
	struct buddy_chunk *chunk = NULL;
	UT_array *ranks;

	// Allocate allocator
	b = malloc(sizeof(*b));
	if (b == NULL)
		return -AML_ENOMEM;

	// Hash table must be initialized to NULL.
	b->allocations = NULL;

	// Allocate ranks list.
	utarray_new(ranks, &ranks_icd);
	if (err == -AML_ENOMEM)
		goto err_with_allocator;
	b->ranks = ranks;

	// Initialize buddy lists to NULL (empty lists).
	for (size_t i = 0; i < MAX_BUDDY_RANK; i++) {
		utarray_push_back(b->ranks, &chunk);
		if (err == -AML_ENOMEM)
			goto err_with_array;
	}

	*out = b;
	return AML_SUCCESS;

err_with_array:
	utarray_free(b->ranks);
err_with_allocator:
	free(b);
	return err;
}

int buddy_allocator_destroy(struct buddy_allocator **buddy)
{
	struct buddy_allocator *b = *buddy;
	struct buddy_chunk *head, *elt, *tmp;

	// Check for remaining allocated blocks.
	HASH_ITER(hh, b->allocations, elt, tmp)
	{
		return -AML_EBUSY;
	}

	// For each buddy clear buddy list.
	for (size_t i = 0; i < utarray_len(b->ranks); i++) {
		// Walk the buddy list and delete chunks.
		head = *(struct buddy_chunk **)utarray_eltptr(b->ranks, i);
		if (head != NULL) {
			DL_FOREACH_SAFE(head, elt, tmp)
			{
				DL_DELETE(head, elt);
				free(elt);
			}
		}
	}

	// Cleanup utarray of ranks
	utarray_free(b->ranks);
	// Cleanup allocator
	free(b);
	*buddy = NULL;
	return AML_SUCCESS;
}

static int buddy_allocator_merge_chunk(struct buddy_allocator *b,
                                       struct buddy_chunk **c)
{
	struct buddy_chunk *chunk = *c;
	size_t rank = buddy_rank(chunk->size);

	// We can't merge because we are the largest chunk size possible.
	if (rank == 0)
		return -AML_EDOM;

	struct buddy_chunk *next, *prev, *left, *right;

	next = chunk->next_buddy;
	prev = chunk->prev_buddy;
	// Merge from left
	if (next != NULL && next->size == chunk->size && !next->is_allocated) {
		left = chunk;
		right = next;
	}
	// Merge from right
	else if (prev != NULL && prev->size == chunk->size &&
	         !prev->is_allocated) {
		left = prev;
		right = chunk;
	}
	// Impossible to merge.
	else {
		return -AML_EINVAL;
	}

	struct buddy_chunk **head = utarray_eltptr(b->ranks, rank);
	struct buddy_chunk **next_head = utarray_eltptr(b->ranks, rank - 1);

	// Remove chunks from their list.
	// The chunk passed in input may come from a free and not be in a list
	// yet.
	if (!left->is_allocated)
		DL_DELETE(*head, left);
	if (!right->is_allocated)
		DL_DELETE(*head, right);

	// Update left chunk to be the merge result chunk.
	left->size += right->size;
	left->next = NULL;
	left->prev = NULL;
	left->next_buddy = right->next_buddy;
	if (left->next_buddy != NULL)
		left->next_buddy->prev_buddy = left;

	// Append resulting chunk to its rank list.
	left->is_allocated = 0;
	DL_APPEND(*next_head, left);

	// Cleanup unused chunk.
	free(right);

	// Return new chunk to user.
	*c = left;
	return AML_SUCCESS;
}

int buddy_allocator_free(struct buddy_allocator *b, void *ptr)
{
	struct buddy_chunk *c;

	// Lookup if the allocation is from this buddy.
	HASH_FIND_PTR(b->allocations, &ptr, c);
	if (c == NULL) // Not a previous allocation.
		return -AML_EINVAL;

	// Remove from hashtable
	HASH_DEL(b->allocations, c);

	// Try to merge chunk with a neighbor buddy
	int err = buddy_allocator_merge_chunk(b, &c);
	// If first merge fails, we have to insert chunk manually.
	if (err != AML_SUCCESS) {
		struct buddy_chunk **head =
		        utarray_eltptr(b->ranks, buddy_rank(c->size));
		c->next = NULL;
		c->prev = NULL;
		c->is_allocated = 0;
		DL_APPEND(*head, c);
	}
	// If first merge succeeds, resulting chunk is in one of the rank lists.
	// we keep merging it until it fails.
	else
		while (err == AML_SUCCESS)
			err = buddy_allocator_merge_chunk(b, &c);

	return AML_SUCCESS;
}

static void buddy_allocator_defragment(struct buddy_allocator *b)
{
	for (size_t i = MAX_BUDDY_RANK - 1; i > 0; i--) {
		int err;
		struct buddy_chunk *chunk;
		struct buddy_chunk **head = utarray_eltptr(b->ranks, i);

		// Merge matching chunks
		chunk = *head;

		while (chunk != NULL) {
			err = buddy_allocator_merge_chunk(b, &chunk);
			if (err == -AML_EINVAL) // chunk is not updated.
				chunk = chunk->next;
			else if (err == AML_SUCCESS) // chunk is moved or freed.
				chunk = *head;
		}
	}
}

// Split a chunk from the next buddy and store it in the current buddy.
// Apply recursively to next buddy if next buddy has no chunk.
static int buddy_allocator_split_next(struct buddy_allocator *b, size_t rank)
{
	int err;

	// We cannot split_next memory from next buddy because we are the last
	// buddy.
	if (rank == 0)
		return -AML_ENOMEM;

	struct buddy_chunk **head = utarray_eltptr(b->ranks, rank);
	struct buddy_chunk **next_head = utarray_eltptr(b->ranks, rank - 1);

	// If there is no memory in next rank list, recursively split_next
	// memory.
	if (*next_head == NULL) {
		err = buddy_allocator_split_next(b, rank - 1);
		if (err != AML_SUCCESS)
			return err;
	}

	// Split was successful, therefore, the list represented by `next_buddy`
	// must not be empty anymore.
	assert(*next_head != NULL);

	struct buddy_chunk *left = *next_head;
	size_t size = left->size / 2;
	struct buddy_chunk *right =
	        buddy_chunk_create((void *)((uintptr_t)left->ptr + size), size,
	                           left->next_buddy, left);
	if (right == NULL)
		return -AML_ENOMEM;
	DL_APPEND(*head, right);

	// Remove from next rank list.
	// Update and append to this rank list
	DL_DELETE(*next_head, left);
	left->size = size;
	left->next = NULL;
	left->prev = NULL;
	DL_APPEND(*head, left);

	return AML_SUCCESS;
}

int buddy_allocator_alloc(struct buddy_allocator *b, void **out, size_t size)
{
	// Make sure size is at least the minimum required size.
	size = size < MIN_SIZE ? MIN_SIZE : size;

	size_t rank = buddy_rank(size);
	// If size is at least MIN_SIZE, it must fit somewhere in ranks array.
	assert(rank < MAX_BUDDY_RANK);

	int err = AML_SUCCESS;
	struct buddy_chunk *chunk, **buddy = utarray_eltptr(b->ranks, rank);

	// If there is a chunk available in matching buddy, return it.
	if (*buddy != NULL)
		goto pop_buddy;

	// If there is no chunk available, we have to look for one in next
	// ranks.
	err = buddy_allocator_split_next(b, rank);
	if (err == AML_SUCCESS)
		goto pop_buddy;

	// If there was no chunk available in next ranks, we try to defragment
	// the allocator to make larger chunks then try again.
	// Defragmentation will concatenate matching chunks and move them to the
	// buddy of the same rank.
	buddy_allocator_defragment(b);

	// A chunk is available in matching buddy.
	if (*buddy != NULL)
		goto pop_buddy;

	// There is at least one chunk in next ranks.
	err = buddy_allocator_split_next(b, rank);
	if (err == AML_SUCCESS)
		goto pop_buddy;

	// We tried our best but there is no memory available.
	return -AML_ENOMEM;

pop_buddy:
	// This is the buddy from which we can pop a chunk.
	chunk = *buddy;
	// Add the chunk to the hashtable of freed chunk.
	err = AML_SUCCESS;
	HASH_ADD_PTR(b->allocations, ptr, chunk);
	if (err == -AML_ENOMEM)
		return -AML_ENOMEM;
	// Mark as allocated
	chunk->is_allocated = 1;
	// Remove chunk from the buddy list
	DL_DELETE(*buddy, chunk);
	// Return chunk pointer.
	*out = chunk->ptr;
	return AML_SUCCESS;
}

//------------------------------------------------------------------------------
// AML Buddy allocator.
//------------------------------------------------------------------------------

static size_t closest_greater_power_of_two(size_t size)
{
	size_t i = 0;

	// Shift left until we meet a bit that is not 0.
	while ((size >> i) != 0)
		i++;
	return 1UL << (i + 1);
}

/** Add a new memory pool to allocator. */
static int aml_allocator_buddy_extend(struct aml_allocator_buddy *data,
                                      size_t size)
{
	int err = AML_SUCCESS;
	void *ptr = NULL;
	struct buddy_chunk *chunk;
	UT_array *pools = (UT_array *)data->pools;
	size_t last_size;

	// Make size a power of two.
	last_size = closest_greater_power_of_two(size);

	// Grow by a size of two compared to previous mapping.
	if (utarray_len(pools) > 0) {
		chunk = *(struct buddy_chunk **)utarray_eltptr(
		        pools, utarray_len(pools) - 1);
		if (last_size < chunk->size * 2)
			last_size = chunk->size * 2;
	}

	// Map memory. If it fails, divide size by two, until it is less
	// than requested size.
	while (last_size >= size &&
	       (ptr = aml_area_mmap(data->area, last_size, data->opts)) == NULL)
		last_size = last_size >> 1;
	if (ptr == NULL)
		return aml_errno;

	// Create and initialize memory pool.
	chunk = buddy_chunk_create(ptr, last_size, NULL, NULL);
	if (chunk == NULL) {
		err = -AML_ENOMEM;
		goto err_with_ptr;
	}

	// Push new pool into the vector of pools.
	utarray_push_back(pools, &chunk);
	if (err != AML_SUCCESS)
		goto err_with_chunk;

	// Extend low-level allocator with new chunk.
	err = buddy_allocator_extend(data->allocator, ptr, last_size);
	if (err != AML_SUCCESS)
		goto err_after_push;

	return AML_SUCCESS;

err_after_push:
	utarray_pop_back(pools);
err_with_chunk:
	free(chunk);
err_with_ptr:
	aml_area_munmap(data->area, ptr, last_size);
	return -AML_ENOMEM;
}

int aml_allocator_buddy_create(struct aml_allocator **allocator,
                               struct aml_area *area,
                               struct aml_area_mmap_options *opts)
{
	if (allocator == NULL || area == NULL)
		return -AML_EINVAL;

	int err = AML_SUCCESS;
	struct aml_allocator *alloc;
	struct aml_allocator_buddy *data;
	UT_array *pools;

	// Allocate high level structure and metadata.
	alloc = AML_INNER_MALLOC(struct aml_allocator,
	                         struct aml_allocator_buddy);
	if (alloc == NULL)
		return -AML_ENOMEM;
	alloc->data = AML_INNER_MALLOC_GET_FIELD(alloc, 2, struct aml_allocator,
	                                         struct aml_allocator_buddy);
	data = (struct aml_allocator_buddy *)alloc->data;
	alloc->ops = &aml_allocator_buddy_ops;

	// Fill metadata
	data->area = area;
	data->opts = opts;

	// Allocate space for pools.
	utarray_new(pools, &ranks_icd);
	if (err != AML_SUCCESS)
		goto err_with_allocator;
	data->pools = pools;

	// Initialize mutex.
	if (pthread_mutex_init(&data->lock, NULL) != 0) {
		err = -AML_FAILURE;
		goto err_with_vec;
	}

	// Allocate buddy_allocator.
	err = buddy_allocator_create(&data->allocator);
	if (err != AML_SUCCESS)
		goto err_with_lock;

	*allocator = alloc;
	return AML_SUCCESS;

err_with_lock:
	pthread_mutex_destroy(&data->lock);
err_with_vec:
	utarray_free(pools);
err_with_allocator:
	free(alloc);
	return err;
}

int aml_allocator_buddy_destroy(struct aml_allocator **allocator)
{
	if (allocator == NULL || *allocator == NULL ||
	    (*allocator)->data == NULL)
		return -AML_EINVAL;

	int err;
	struct aml_allocator_buddy *data =
	        (struct aml_allocator_buddy *)(*allocator)->data;
	struct buddy_chunk *chunk;
	UT_array *pools = data->pools;

	// Be the last to lock the mutex.
	pthread_mutex_lock(&data->lock);

	// Delete buddy allocator
	// If some allocations are still in use, this will fail.
	// data->allocator may be NULL if previous destroy() call failed after
	// this step.
	if (data->allocator != NULL) {
		err = buddy_allocator_destroy(&data->allocator);
		if (err != AML_SUCCESS)
			goto error;
	}

	// Unmap memory pools
	size_t len;
	while ((len = utarray_len(pools)) > 0) {
		chunk = *(struct buddy_chunk **)utarray_eltptr(pools, len - 1);
		err = aml_area_munmap(data->area, chunk->ptr, chunk->size);
		// If munmap() fails here, the allocator is in an inconsistent
		// state. It can only be used to be destroyed.
		if (err != AML_SUCCESS)
			goto error;
		utarray_pop_back(pools);
		free(chunk);
	}
	utarray_free(pools);

	// Cleanup top level structure.
	pthread_mutex_unlock(&data->lock); // Necessary to avoid UB on destroy.
	pthread_mutex_destroy(&data->lock);
	free(*allocator);
	*allocator = NULL;

	return AML_SUCCESS;
error:
	pthread_mutex_unlock(&data->lock);
	return err;
}

void *aml_allocator_buddy_alloc(struct aml_allocator_data *data, size_t size)
{
	int err;
	void *ptr;
	struct aml_allocator_buddy *alloc = (struct aml_allocator_buddy *)data;

	pthread_mutex_lock(&alloc->lock);

	// Try to pop a chunk from the low-level allocator.
	err = buddy_allocator_alloc(alloc->allocator, &ptr, size);
	if (err == AML_SUCCESS)
		goto success;

	// Low-level allocator is out of memory.
	// We map more memory and try again.
	err = aml_allocator_buddy_extend(alloc, size);
	if (err != AML_SUCCESS)
		goto error;

	// Get a chunk from the new pool.
	err = buddy_allocator_alloc(alloc->allocator, &ptr, size);
	// Pool extension worked, therefore, there must be enough memory
	// to satisfy allocation.
	assert(err == AML_SUCCESS);

success:
	pthread_mutex_unlock(&alloc->lock);
	return ptr;

error:
	pthread_mutex_unlock(&alloc->lock);
	aml_errno = err;
	return NULL;
}

int aml_allocator_buddy_free(struct aml_allocator_data *data, void *ptr)
{
	struct aml_allocator_buddy *alloc = (struct aml_allocator_buddy *)data;

	int err;
	pthread_mutex_lock(&alloc->lock);
	err = buddy_allocator_free(alloc->allocator, ptr);
	pthread_mutex_unlock(&alloc->lock);
	return err;
}

struct aml_allocator_ops aml_allocator_buddy_ops = {
        .alloc = aml_allocator_buddy_alloc,
        .free = aml_allocator_buddy_free,
};
