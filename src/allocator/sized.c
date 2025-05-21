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

#include "internal/uthash.h"
#include "internal/utlist.h"

#define PTR_OFF(ptr, sign, off) (void *)((intptr_t)(ptr)sign(intptr_t)(off))
#define ALIGNED_PTR(ptr, s) PTR_OFF(ptr, +, s - ((intptr_t)ptr % s))

struct aml_sized_chunk {
	void *ptr;
	struct aml_sized_chunk *prev, *next;
	UT_hash_handle hh;
};

/** Sized allocator metadata. */
struct aml_allocator_sized {
    /** c inheritance */
    struct aml_allocator super;
    /** The size of user allocations. */
    size_t chunk_size;
    /** The free mapped memory regions (internally a utlist) */
    struct aml_sized_chunk *free_pools;
    /** The free mapped memory regions (internally a uthash) */
    struct aml_sized_chunk *occupied_pools;
};

/** Add a new memory chunk to allocator. */
static int aml_allocator_sized_extend(struct aml_allocator_sized *alloc, size_t size)
{
	void *ptr;
	struct aml_sized_chunk *chunk = NULL;

	ptr = aml_area_mmap(alloc->super.area, size, alloc->super.opts);
	if (ptr == NULL)
		return aml_errno;

	chunk = malloc(sizeof(*chunk));
	if (chunk == NULL)
		return -AML_ENOMEM;

	chunk->ptr = ptr;
	DL_APPEND(alloc->free_pools, chunk);
	return AML_SUCCESS;
}

void *aml_allocator_sized_alloc(struct aml_allocator *allocator, size_t size)
{
	int err;
	struct aml_allocator_sized *alloc = (struct aml_allocator_sized *) allocator;
	struct aml_sized_chunk *ret = NULL;

	// Check input
	if (size != alloc->chunk_size) {
		aml_errno = -AML_EINVAL;
		return NULL;
	}

	if (alloc->free_pools == NULL) {
		err = aml_allocator_sized_extend((struct aml_allocator_sized *)alloc, size);
		if (err != AML_SUCCESS) {
			aml_errno = err;
			return NULL;
		}
	}

	/* pop element */
	ret = alloc->free_pools;
	DL_DELETE(alloc->free_pools, ret);

	/* put it in the occupied hash */
	HASH_ADD_PTR(alloc->occupied_pools, ptr, ret);

	return ret->ptr;
}

int aml_allocator_sized_free(struct aml_allocator *allocator, void *ptr)
{
	struct aml_allocator_sized *alloc = (struct aml_allocator_sized *) allocator;
	struct aml_sized_chunk *elt = NULL;

	HASH_FIND_PTR(alloc->occupied_pools, &ptr, elt);
	if (elt == NULL)
		return -AML_EINVAL;

	HASH_DEL(alloc->occupied_pools, elt);
	DL_APPEND(alloc->free_pools, elt);
	return AML_SUCCESS;
}

struct aml_allocator_ops aml_allocator_sized_ops = {
        .alloc = aml_allocator_sized_alloc,
        .free = aml_allocator_sized_free,
        .give = NULL,
        .alloc_chunk = NULL,
        .free_chunk = NULL,
};

int aml_allocator_sized_create(struct aml_allocator **allocator,
                               size_t size,
                               struct aml_area *area,
                               struct aml_area_mmap_options *opts)
{
    AML_ALLOCATOR_CREATE_BEGIN(allocator, area, opts, &aml_allocator_sized_ops, struct aml_allocator_sized, alloc)
    {
        int err = aml_allocator_sized_extend(alloc, size);
        if (err != AML_SUCCESS)
            AML_ALLOCATOR_CREATE_FAIL(allocator, area, opts, &aml_allocator_sized_ops, struct aml_allocator_sized, alloc);
    }
    AML_ALLOCATOR_CREATE_END(allocator, area, opts, &aml_allocator_sized_ops, struct aml_allocator_sized, alloc);
}

int aml_allocator_sized_destroy(struct aml_allocator **allocator)
{
    AML_ALLOCATOR_DESTROY_BEGIN(allocator, struct aml_allocator_sized, alloc)
    {
        struct aml_sized_chunk *cur, *tmp;

        HASH_ITER(hh, alloc->occupied_pools, cur, tmp)
        {
            aml_area_munmap(alloc->super.area, cur->ptr, alloc->chunk_size);
            HASH_DEL(alloc->occupied_pools, cur);
            free(cur);
        }

        DL_FOREACH_SAFE(alloc->free_pools, cur, tmp)
        {
            aml_area_munmap(alloc->super.area, cur->ptr, alloc->chunk_size);
            DL_DELETE(alloc->free_pools, cur);
            free(cur);
        }
    }
    AML_ALLOCATOR_DESTROY_END(allocator, struct aml_allocator_sized, alloc);
}
