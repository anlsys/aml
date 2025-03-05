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

// TODO : currently, max rank represent 2^(64-6) chunk (=256 PB)
// We can probably reduce to 2^(40) bytes chunk (=1 TB)
// so static loops over 'MAX_BUDDY_RANK' gets faster

// The smallest chunk index in ranks array.
// The corresponding size is 2^6 = 64 bytes.
// The allocator won't accept allocations smaller than this.
#define MIN_BUDDY_RANK (6)
#define MAX_BUDDY_RANK ((sizeof(size_t) * 8) - MIN_BUDDY_RANK)
#define MIN_SIZE (1 << MIN_BUDDY_RANK)

struct buddy_chunk {
    // aml abstract chunk inheritance
	struct aml_allocator_chunk super;
    // uthash handle
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

// Return 2^n so that 2^{n-1} < size <= 2^n
static inline size_t closest_greater_power_of_two(size_t size)
{
    size_t i = 0;
    while ((1UL << i) < size)
        ++i;
    return (1UL << i);
}

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
    assert(size >= MIN_SIZE);

    int is_power2 = is_power_of_two(size);

    // retrieve index of the last bit set
    int i = 0;
    while ((size >> (MIN_BUDDY_RANK+i)) > 1)
        ++i;

    if (!is_power_of_two(size))
        ++i;

    return MAX_BUDDY_RANK - i - 1;
}

static struct buddy_chunk *buddy_chunk_create(void *ptr,
                                              size_t size,
                                              struct buddy_chunk *next_buddy,
                                              struct buddy_chunk *prev_buddy)
{
	struct buddy_chunk *chunk = malloc(sizeof(*chunk));
	if (chunk == NULL)
		return NULL;
	chunk->super.ptr = ptr;
	chunk->super.size = size;
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

	if (l->super.ptr == r->super.ptr)
		return 0;
	else if (l->super.ptr < r->super.ptr)
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
	// of size 2^(MAX_BUDDY_RANK-index).
    struct buddy_chunk *ranks[MAX_BUDDY_RANK];
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

	buddy = b->ranks + rank;
	DL_APPEND(*buddy, chunk);

	return AML_SUCCESS;
}

/** Add a new memory pool to allocator. */
static int aml_allocator_buddy_extend(struct aml_allocator_buddy *data,
                                      size_t size)
{
	int err = AML_SUCCESS;
	void *ptr;
	struct buddy_chunk *chunk;
	UT_array *pools = (UT_array *)data->pools;

	size_t last_size = closest_greater_power_of_two(size);

    // Grow by a size of two compared to previous mapping.
	if (utarray_len(pools) > 0) {
		chunk = *(struct buddy_chunk **)utarray_eltptr(pools, utarray_len(pools) - 1);
        assert(chunk);
		if (last_size < chunk->super.size * 2)
			last_size = chunk->super.size * 2;
	}

	// Map memory. If it fails, divide size by two, until it is less
	// than requested size.
	while (1)
    {
        ptr = aml_area_mmap(data->area, last_size, data->opts);
        if (ptr)
            break ;
        last_size = last_size >> 1;
        if (last_size < size)
		    return -AML_ENOMEM;
    }
    assert(ptr);

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


int buddy_allocator_create(struct buddy_allocator **out)
{
	int err = AML_SUCCESS;
	struct buddy_allocator *b;

	// Allocate allocator
	b = malloc(sizeof(*b));
	if (b == NULL)
		return -AML_ENOMEM;

	// Hash table must be initialized to NULL.
	b->allocations = NULL;

	// Initialize buddy lists to NULL (empty lists).
	memset(b->ranks, 0, sizeof(b->ranks));

	*out = b;
	return AML_SUCCESS;
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
	for (size_t i = 0; i < MAX_BUDDY_RANK; i++) {
		// Walk the buddy list and delete chunks.
		head = b->ranks[i];
		if (head != NULL) {
			DL_FOREACH_SAFE(head, elt, tmp)
			{
				DL_DELETE(head, elt);
				free(elt);
			}
		}
	}

	// Cleanup allocator
	free(b);
	*buddy = NULL;
	return AML_SUCCESS;
}

static int buddy_allocator_merge_chunk(struct buddy_allocator *b,
                                       struct buddy_chunk **c)
{
	struct buddy_chunk *chunk = *c;
	size_t rank = buddy_rank(chunk->super.size);

	// We can't merge because we are the largest chunk size possible.
	if (rank == 0)
		return -AML_EDOM;

	struct buddy_chunk *next, *prev, *left, *right;

	next = chunk->next_buddy;
	prev = chunk->prev_buddy;
	// Merge from left
	if (next != NULL && next->super.size == chunk->super.size && !next->is_allocated) {
		left = chunk;
		right = next;
	}
	// Merge from right
	else if (prev != NULL && prev->super.size == chunk->super.size &&
	         !prev->is_allocated) {
		left = prev;
		right = chunk;
	}
	// Impossible to merge.
	else {
		return -AML_EINVAL;
	}

    assert(rank > 0);
	struct buddy_chunk **head      = b->ranks + rank;
	struct buddy_chunk **next_head = b->ranks + (rank - 1);

	// Remove chunks from their list.
	// The chunk passed in input may come from a free and not be in a list
	// yet.
	if (!left->is_allocated)
		DL_DELETE(*head, left);
	if (!right->is_allocated)
		DL_DELETE(*head, right);

	// Update left chunk to be the merge result chunk.
	left->super.size += right->super.size;
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

static void buddy_allocator_defragment(struct buddy_allocator *b)
{
	for (size_t i = MAX_BUDDY_RANK - 1; i > 0; i--) {
		int err;
		struct buddy_chunk *chunk;
		struct buddy_chunk **head = b->ranks + i;

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

	struct buddy_chunk **head = b->ranks + rank;
	struct buddy_chunk **next_head = b->ranks + (rank - 1);

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
	size_t size = left->super.size / 2;
	struct buddy_chunk *right =
	        buddy_chunk_create((void *)((uintptr_t)left->super.ptr + size), size,
	                           left->next_buddy, left);
	if (right == NULL)
		return -AML_ENOMEM;
	DL_APPEND(*head, right);

	// Remove from next rank list.
	// Update and append to this rank list
	DL_DELETE(*next_head, left);
	left->super.size = size;
	left->next = NULL;
	left->prev = NULL;
	DL_APPEND(*head, left);

	return AML_SUCCESS;
}

static int buddy_alloc_chunk_do(struct buddy_allocator *b, struct buddy_chunk **out, size_t size, int save_hash)
{
	size_t rank = buddy_rank(size);
	// If size is at least MIN_SIZE, it must fit somewhere in ranks array.
	assert(rank < MAX_BUDDY_RANK);

	int err = AML_SUCCESS;
	struct buddy_chunk *chunk;
    struct buddy_chunk **buddy = b->ranks + rank;

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
    if (save_hash)
    {
	    HASH_ADD_PTR(b->allocations, super.ptr, chunk);
	    if (err == -AML_ENOMEM)
		    return -AML_ENOMEM;
    }
	// Mark as allocated
	chunk->is_allocated = 1;
	// Remove chunk from the buddy list
	DL_DELETE(*buddy, chunk);
	// Return chunk pointer.
	*out = chunk;
	return AML_SUCCESS;
}

struct aml_allocator_chunk *buddy_alloc_chunk(struct aml_allocator_data *data, size_t size, int save_hash)
{
    struct aml_allocator_buddy *alloc = (struct aml_allocator_buddy *) data;
    struct buddy_chunk *c;

    // Make sure size is at least the minimum required size.
    size = size < MIN_SIZE ? MIN_SIZE : size;

	pthread_mutex_lock(&alloc->lock);
    {
        // Try to pop a chunk from the low-level allocator.
        int err = buddy_alloc_chunk_do(alloc->allocator, &c, size, save_hash);
        if (err != AML_SUCCESS)
        {
            // Low-level allocator is out of memory.
            // We map more memory and try again.
            err = aml_allocator_buddy_extend(alloc, size);
            if (err == AML_SUCCESS)
            {
                // Get a chunk from the new pool.
                err = buddy_alloc_chunk_do(alloc->allocator, &c, size, save_hash);
                // Pool extension worked, therefore, there must be enough memory
                // to satisfy allocation.
                assert(err == AML_SUCCESS);
            }
            else
            {
                aml_errno = err;
                c = NULL;
            }
        }
    }
	pthread_mutex_unlock(&alloc->lock);

    return (struct aml_allocator_chunk *) c;
}

//------------------------------------------------------------------------------
// AML Buddy allocator.
//------------------------------------------------------------------------------

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
		err = aml_area_munmap(data->area, chunk->super.ptr, chunk->super.size);
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

struct aml_allocator_chunk *aml_allocator_buddy_alloc_chunk(struct aml_allocator_data *data, size_t size)
{
    return buddy_alloc_chunk(data, size, 0);
}

void *aml_allocator_buddy_alloc(struct aml_allocator_data *data, size_t size)
{
    struct aml_allocator_chunk *c = buddy_alloc_chunk(data, size, 1);
	return c ? c->ptr : NULL;
}

int buddy_free_chunk(struct buddy_allocator *b, struct buddy_chunk *c)
{
	// Try to merge chunk with a neighbor buddy
	int err = buddy_allocator_merge_chunk(b, &c);
	// If first merge fails, we have to insert chunk manually.
	if (err != AML_SUCCESS) {
        int rank = buddy_rank(c->super.size);
		struct buddy_chunk **head = b->ranks + rank;
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

int aml_allocator_buddy_free_chunk(struct aml_allocator_data *data, struct aml_allocator_chunk *c)
{
	struct aml_allocator_buddy *alloc = (struct aml_allocator_buddy *)data;

	int err;
	pthread_mutex_lock(&alloc->lock);
    {
        err = buddy_free_chunk(alloc->allocator, (struct buddy_chunk *) c);
    }
    pthread_mutex_unlock(&alloc->lock);
	return err;
}

int aml_allocator_buddy_free(struct aml_allocator_data *data, void *ptr)
{
	struct aml_allocator_buddy *alloc = (struct aml_allocator_buddy *)data;

	int err;
	pthread_mutex_lock(&alloc->lock);
    {
        struct buddy_allocator *b = alloc->allocator;
        struct buddy_chunk *c;

        // Lookup if the allocation is from this buddy.
        HASH_FIND_PTR(b->allocations, &ptr, c);
        if (c == NULL) // Not a previous allocation.
            err = -AML_EINVAL;
        else
        {
            // Remove from hashtable
            HASH_DEL(b->allocations, c);

            err = buddy_free_chunk(alloc->allocator, c);
        }
    }
    pthread_mutex_unlock(&alloc->lock);
	return err;
}

struct aml_allocator_ops aml_allocator_buddy_ops = {
        .alloc = aml_allocator_buddy_alloc,
        .free = aml_allocator_buddy_free,
        .alloc_chunk = aml_allocator_buddy_alloc_chunk,
        .free_chunk = aml_allocator_buddy_free_chunk
};
