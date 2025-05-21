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
#include "aml/higher/allocator/area.h"

#include "internal/uthash.h"

struct aml_allocator_ops aml_allocator_area_ops = {
        .alloc = aml_allocator_area_alloc,
        .free = aml_allocator_area_free,
        .give = NULL,
        .alloc_chunk = NULL,
        .free_chunk = NULL};

int aml_allocator_area_create(struct aml_allocator **out,
                              struct aml_area *area,
                              struct aml_area_mmap_options *opts)
{
    AML_ALLOCATOR_CREATE_BEGIN(out, area, opts, &aml_allocator_area_ops, struct aml_allocator_area, alloc)
    {
        alloc->chunks = NULL;
    }
    AML_ALLOCATOR_CREATE_END(out, area, opts, &aml_allocator_area_ops, struct aml_allocator_area, alloc);
}

int aml_allocator_area_destroy(struct aml_allocator **allocator)
{
    AML_ALLOCATOR_DESTROY_BEGIN(allocator, struct aml_allocator_area, alloc)
    {
        struct aml_allocator_area_chunk *current, *tmp;
        HASH_ITER(hh, alloc->chunks, current, tmp)
        {
            HASH_DEL(alloc->chunks, current);
            int err = aml_area_munmap(alloc->super.area, current->ptr, current->size);
            if (err != AML_SUCCESS)
            {
                HASH_ADD_PTR(alloc->chunks, ptr, current);
                AML_ALLOCATOR_DESTROY_FAIL(allocator, struct aml_allocator_area, alloc);
            }
            free(current);
        }
    }
    AML_ALLOCATOR_DESTROY_END(allocator, struct aml_allocator_area, alloc);
}

void *aml_allocator_area_alloc(struct aml_allocator *allocator, size_t size)
{
	struct aml_allocator_area *alloc = (struct aml_allocator_area *) allocator;
	struct aml_allocator_area_chunk *chunk;

	// Create chunk info to store in hashtable.
	chunk = malloc(sizeof(*chunk));
	if (chunk == NULL) {
		errno = AML_ENOMEM;
		return NULL;
	}
	chunk->size = size;
	chunk->ptr = aml_area_mmap(allocator->area, size, allocator->opts);

	// Check if mmap worked.
	if (chunk->ptr == NULL) {
		free(chunk);
		return NULL;
	}

	// Add chunk info to hashtable.
	pthread_mutex_lock(&allocator->lock);
	HASH_ADD_PTR(alloc->chunks, ptr, chunk);
	pthread_mutex_unlock(&allocator->lock);

	return chunk->ptr;
}

int aml_allocator_area_free(struct aml_allocator *allocator, void *ptr)
{
	struct aml_allocator_area *alloc = (struct aml_allocator_area *) allocator;
	struct aml_allocator_area_chunk *chunk;
	int err = AML_SUCCESS;

	// Lookup ptr in this allocator hashtable.
	pthread_mutex_lock(&allocator->lock);
	HASH_FIND_PTR(alloc->chunks, &ptr, chunk);
	if (chunk == NULL) {
		err = -AML_EINVAL;
		goto out;
	}

	// Unmap pointer
	err = aml_area_munmap(allocator->area, ptr, chunk->size);
	if (err != AML_SUCCESS)
		goto out;

	// Remove from hashtable
	HASH_DEL(alloc->chunks, chunk);
	free(chunk);
out:
	pthread_mutex_unlock(&allocator->lock);
	return err;
}
