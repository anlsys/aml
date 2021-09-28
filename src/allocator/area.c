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
};

struct aml_allocator_area_chunk {
	void *ptr;
	size_t size;
	UT_hash_handle hh;
};

int aml_allocator_area_create(struct aml_allocator **out,
                              struct aml_area *area,
                              struct aml_area_mmap_options *opts)
{
	if (out == NULL || area == NULL)
		return -AML_EINVAL;

	struct aml_allocator *alloc = AML_INNER_MALLOC(
	        struct aml_allocator, struct aml_allocator_area_data);

	if (alloc == NULL)
		return -AML_ENOMEM;

	alloc->data = AML_INNER_MALLOC_GET_FIELD(
	        alloc, 2, struct aml_allocator, struct aml_allocator_area_data);
	alloc->ops = &aml_allocator_area_ops;

	struct aml_allocator_area_data *data =
	        (struct aml_allocator_area_data *)alloc->data;
	data->area = area;
	data->opts = opts;
	data->chunks = NULL;

	if (pthread_mutex_init(&data->lock, NULL) != 0) {
		free(alloc);
		return -AML_FAILURE;
	}

	*out = alloc;
	return AML_SUCCESS;
}

#define CHUNKS(data) (struct aml_allocator_area_chunk *)data->chunks

int aml_allocator_area_destroy(struct aml_allocator **allocator)
{
	if (allocator == NULL || *allocator == NULL)
		return -AML_EINVAL;

	int err;
	struct aml_allocator_area_data *data;
	struct aml_allocator_area_chunk *current, *tmp, *chunks;

	data = (struct aml_allocator_area_data *)(*allocator)->data;
	chunks = CHUNKS(data);

	// Be the last to lock the mutex.
	pthread_mutex_lock(&data->lock);

	HASH_ITER(hh, chunks, current, tmp)
	{
		HASH_DEL(chunks, current);
		err = aml_area_munmap(data->area, current->ptr, current->size);
		if (err != AML_SUCCESS) {
			HASH_ADD_PTR(chunks, ptr, current);
			data->chunks = (void *)chunks;
			return err;
		}
		free(current);
	}

	// Destroy mutex.
	if (pthread_mutex_destroy(&data->lock) != 0)
		return -AML_FAILURE;

	// Free allocator.
	free(*allocator);
	*allocator = NULL;
	return AML_SUCCESS;
}

void *aml_allocator_area_alloc(struct aml_allocator_data *data, size_t size)
{
	struct aml_allocator_area_data *d =
	        (struct aml_allocator_area_data *)data;
	struct aml_allocator_area_chunk *chunk, *chunks = CHUNKS(d);

	// Create chunk info to store in hashtable.
	chunk = malloc(sizeof(*chunk));
	if (chunk == NULL) {
		errno = AML_ENOMEM;
		return NULL;
	}
	chunk->size = size;
	chunk->ptr = aml_area_mmap(d->area, size, d->opts);

	// Check if mmap worked.
	if (chunk->ptr == NULL) {
		free(chunk);
		return NULL;
	}

	// Add chunk info to hashtable.
	pthread_mutex_lock(&d->lock);
	HASH_ADD_PTR(chunks, ptr, chunk);
	d->chunks = chunks;
	pthread_mutex_unlock(&d->lock);

	return chunk->ptr;
}

int aml_allocator_area_free(struct aml_allocator_data *data, void *ptr)
{
	struct aml_allocator_area_data *d =
	        (struct aml_allocator_area_data *)data;
	struct aml_allocator_area_chunk *chunk, *chunks = CHUNKS(d);
	int err = AML_SUCCESS;

	// Lookup ptr in this allocator hashtable.
	pthread_mutex_lock(&d->lock);
	HASH_FIND_PTR(chunks, &ptr, chunk);
	if (chunk == NULL) {
		err = -AML_EINVAL;
		goto out;
	}

	// Unmap pointer
	err = aml_area_munmap(d->area, ptr, chunk->size);
	if (err != AML_SUCCESS)
		goto out;

	// Remove from hashtable
	HASH_DEL(chunks, chunk);
	d->chunks = chunks;
	free(chunk);
out:
	pthread_mutex_unlock(&d->lock);
	return err;
}
