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
#include "aml/scratch/seq.h"
#include <assert.h>

/*******************************************************************************
 * Sequential scratchpad
 * The scratch itself is organized into several different components
 * - request types: push and pull
 * - implementation of the request
 * - user API (i.e. generic request creation and call)
 * - how to init the scratch
 ******************************************************************************/

/*******************************************************************************
 * Requests:
 ******************************************************************************/
int aml_scratch_request_seq_create(struct aml_scratch_request_seq **req,
				   int uuid)
{
	assert(req != NULL);
	*req = calloc(1, sizeof(struct aml_scratch_request_seq));
	if (*req == NULL)
		return -AML_ENOMEM;
	(*req)->uuid = uuid;
	return 0;
}

void aml_scratch_request_seq_destroy(struct aml_scratch_request_seq **req)
{
	assert(req != NULL);
	free(*req);
	*req = NULL;
}

void aml_scratch_seq_request_data_init(struct aml_scratch_seq_request_data *req,
				       int type, struct aml_tiling *t,
				       void *dstptr, int dstid,
				       void *srcptr, int srcid)
{
	assert(req != NULL);
	req->type = type;
	req->tiling = t;
	req->srcptr = srcptr;
	req->srcid = srcid;
	req->dstptr = dstptr;
	req->dstid = dstid;
}

/*******************************************************************************
 * Internal functions
 ******************************************************************************/
int aml_scratch_seq_doit(struct aml_scratch_seq_data *scratch,
			      struct aml_scratch_seq_request_data *req)
{
	assert(scratch != NULL);
	assert(req != NULL);
	void *dest, *src;
	size_t size;

	dest = aml_tiling_tilestart(req->tiling, req->dstptr, req->dstid);
	src = aml_tiling_tilestart(req->tiling, req->srcptr, req->srcid);
	size = aml_tiling_tilesize(req->tiling, req->srcid);
	return aml_dma_async_copy(scratch->dma, &req->dma_req,
				  AML_DMA_REQUEST_TYPE_PTR, dest, src, size);
}

struct aml_scratch_seq_ops aml_scratch_seq_inner_ops = {
	aml_scratch_seq_doit,
};

/*******************************************************************************
 * Public API
 ******************************************************************************/

int aml_scratch_seq_create_request(struct aml_scratch_data *d,
				   struct aml_scratch_request **r,
				   int type, va_list ap)
{
	assert(d != NULL);
	assert(r != NULL);
	struct aml_scratch_seq *scratch =
		(struct aml_scratch_seq *)d;
	struct aml_scratch_request_seq *ret;
	struct aml_scratch_seq_request_data *req;

	pthread_mutex_lock(&scratch->data.lock);
	req = aml_vector_add(scratch->data.requests);

	/* init the request */
	if (type == AML_SCRATCH_REQUEST_TYPE_PUSH) {
		int scratchid;
		int *srcid;
		void *srcptr;
		void *scratchptr;

		srcptr = va_arg(ap, void *);
		srcid = va_arg(ap, int *);
		scratchptr = va_arg(ap, void *);
		scratchid = va_arg(ap, int);

		/* find destination tile */
		int *slot = aml_vector_get(scratch->data.tilemap, scratchid);

		assert(slot != NULL);
		*srcid = *slot;

		/* init request */
		aml_scratch_seq_request_data_init(req, type,
					     scratch->data.tiling,
					     srcptr, *srcid,
					     scratchptr, scratchid);
	} else if (type == AML_SCRATCH_REQUEST_TYPE_PULL) {
		int *scratchid;
		int srcid;
		void *srcptr;
		void *scratchptr;
		int slot, *tile;

		scratchptr = va_arg(ap, void *);
		scratchid = va_arg(ap, int *);
		srcptr = va_arg(ap, void *);
		srcid = va_arg(ap, int);

		/* find destination tile
		 * We don't use add here because adding a tile means allocating
		 * new tiles on the sch_area too. */
		/* TODO: this is kind of a bug: we reuse a tile, instead of
		 * creating a no-op request
		 */
		slot = aml_vector_find(scratch->data.tilemap, srcid);
		if (slot == -1) {
			slot = aml_vector_find(scratch->data.tilemap, -1);
			assert(slot != -1);
			tile = aml_vector_get(scratch->data.tilemap, slot);
			*tile = srcid;
		} else
			type = AML_SCRATCH_REQUEST_TYPE_NOOP;

		/* save the key */
		*scratchid = slot;

		/* init request */
		aml_scratch_seq_request_data_init(req, type,
						  scratch->data.tiling,
						  scratchptr, *scratchid,
						  srcptr, srcid);
	}
	int uuid = aml_vector_getid(scratch->data.requests, req);

	assert(uuid != AML_SCRATCH_REQUEST_TYPE_INVALID);
	aml_scratch_request_seq_create(&ret, uuid);
	*r = (struct aml_scratch_request *)ret;
	pthread_mutex_unlock(&scratch->data.lock);
	if (req->type != AML_SCRATCH_REQUEST_TYPE_NOOP)
		scratch->ops.doit(&scratch->data, req);
	return 0;
}

int aml_scratch_seq_destroy_request(struct aml_scratch_data *d,
					 struct aml_scratch_request *r)
{
	assert(d != NULL);
	assert(r != NULL);
	struct aml_scratch_seq *scratch =
		(struct aml_scratch_seq *)d;
	struct aml_scratch_seq_request_data *inner_req;
	struct aml_scratch_request_seq *req =
		(struct aml_scratch_request_seq *)r;
	int *tile;

	inner_req = aml_vector_get(scratch->data.requests, req->uuid);
	if (inner_req == NULL)
		return -AML_EINVAL;

	if (inner_req->type != AML_SCRATCH_REQUEST_TYPE_NOOP)
		aml_dma_cancel(scratch->data.dma, inner_req->dma_req);

	/* destroy removes the tile from the scratch */
	if (inner_req->type == AML_SCRATCH_REQUEST_TYPE_PUSH)
		tile = aml_vector_get(scratch->data.tilemap, inner_req->srcid);
	else if (inner_req->type == AML_SCRATCH_REQUEST_TYPE_PULL)
		tile = aml_vector_get(scratch->data.tilemap, inner_req->dstid);
	pthread_mutex_lock(&scratch->data.lock);
	aml_vector_remove(scratch->data.tilemap, tile);
	aml_vector_remove(scratch->data.requests, inner_req);
	pthread_mutex_unlock(&scratch->data.lock);
	aml_scratch_request_seq_destroy(&req);
	return 0;
}

int aml_scratch_seq_wait_request(struct aml_scratch_data *d,
				   struct aml_scratch_request *r)
{
	assert(d != NULL);
	assert(r != NULL);
	struct aml_scratch_seq *scratch = (struct aml_scratch_seq *)d;
	struct aml_scratch_request_seq *req =
		(struct aml_scratch_request_seq *)r;
	struct aml_scratch_seq_request_data *inner_req;
	int *tile;

	inner_req = aml_vector_get(scratch->data.requests, req->uuid);
	if (inner_req == NULL)
		return -AML_EINVAL;

	/* wait for completion of the request */
	if (inner_req->type != AML_SCRATCH_REQUEST_TYPE_NOOP)
		aml_dma_wait(scratch->data.dma, inner_req->dma_req);

	/* cleanup a completed request. In case of push, free up the tile */
	pthread_mutex_lock(&scratch->data.lock);
	if (inner_req->type == AML_SCRATCH_REQUEST_TYPE_PUSH) {
		tile = aml_vector_get(scratch->data.tilemap, inner_req->srcid);
		aml_vector_remove(scratch->data.tilemap, tile);
	}
	aml_vector_remove(scratch->data.requests, inner_req);
	pthread_mutex_unlock(&scratch->data.lock);
	aml_scratch_request_seq_destroy(&req);
	return 0;
}

void *aml_scratch_seq_baseptr(const struct aml_scratch_data *d)
{
	assert(d != NULL);
	const struct aml_scratch_seq *scratch =
		(const struct aml_scratch_seq *)d;

	return scratch->data.sch_ptr;
}

int aml_scratch_seq_release(struct aml_scratch_data *d, int scratchid)
{
	assert(d != NULL);
	struct aml_scratch_seq *scratch = (struct aml_scratch_seq *)d;
	int *tile;

	pthread_mutex_lock(&scratch->data.lock);
	tile = aml_vector_get(scratch->data.tilemap, scratchid);
	if (tile != NULL)
		aml_vector_remove(scratch->data.tilemap, tile);
	pthread_mutex_unlock(&scratch->data.lock);
	return 0;
}

struct aml_scratch_ops aml_scratch_seq_ops = {
	aml_scratch_seq_create_request,
	aml_scratch_seq_destroy_request,
	aml_scratch_seq_wait_request,
	aml_scratch_seq_baseptr,
	aml_scratch_seq_release,
};

/*******************************************************************************
 * Init functions:
 ******************************************************************************/

int aml_scratch_seq_create(struct aml_scratch **scratch,
			   struct aml_area *scratch_area,
			   struct aml_area *src_area,
			   struct aml_dma *dma, struct aml_tiling *tiling,
			   size_t nbtiles, size_t nbreqs)
{
	struct aml_scratch *ret = NULL;
	struct aml_scratch_seq *s;

	if (scratch == NULL
	    || scratch_area == NULL || src_area == NULL
	    || dma == NULL || tiling == NULL)
		return -AML_EINVAL;

	*scratch = NULL;

	ret = AML_INNER_MALLOC_2(struct aml_scratch, struct aml_scratch_seq);
	if (ret == NULL)
		return -AML_ENOMEM;

	ret->ops = &aml_scratch_seq_ops;
	ret->data = AML_INNER_MALLOC_NEXTPTR(ret, struct aml_scratch,
					     struct aml_scratch_seq);
	s = (struct aml_scratch_seq *)ret->data;
	s->ops = aml_scratch_seq_inner_ops;

	s->data.sch_area = scratch_area;
	s->data.src_area = src_area;
	s->data.dma = dma;
	s->data.tiling = tiling;

	/* allocate request array */
	aml_vector_create(&s->data.requests, nbreqs,
			  sizeof(struct aml_scratch_seq_request_data),
			  offsetof(struct aml_scratch_seq_request_data, type),
			  AML_SCRATCH_REQUEST_TYPE_INVALID);

	/* s init */
	aml_vector_create(&s->data.tilemap, nbtiles, sizeof(int), 0, -1);
	size_t tilesize = aml_tiling_tilesize(s->data.tiling, 0);

	s->data.scratch_size = nbtiles * tilesize;
	s->data.sch_ptr = aml_area_mmap(s->data.sch_area,
					      NULL,
					      s->data.scratch_size);
	pthread_mutex_init(&s->data.lock, NULL);

	*scratch = ret;
	return 0;
}

void aml_scratch_seq_destroy(struct aml_scratch **scratch)
{
	struct aml_scratch *s;
	struct aml_scratch_seq *inner;

	if (scratch == NULL)
		return;
	s = *scratch;
	if (s == NULL)
		return;

	assert(s->data != NULL);
	inner = (struct aml_scratch_seq *)s->data;
	aml_vector_destroy(&inner->data.requests);
	aml_vector_destroy(&inner->data.tilemap);
	aml_area_munmap(inner->data.sch_area,
			inner->data.sch_ptr,
			inner->data.scratch_size);
	pthread_mutex_destroy(&inner->data.lock);
	free(s);
	*scratch = NULL;
}
