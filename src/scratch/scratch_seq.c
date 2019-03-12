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

int aml_scratch_request_seq_init(struct aml_scratch_request_seq *req, int type,
				 struct aml_tiling *t, void *dstptr, int dstid,
				 void *srcptr, int srcid)

{
	assert(req != NULL);
	req->type = type;
	req->tiling = t;
	req->srcptr = srcptr;
	req->srcid = srcid;
	req->dstptr = dstptr;
	req->dstid = dstid;
	return 0;
}

int aml_scratch_request_seq_destroy(struct aml_scratch_request_seq *r)
{
	assert(r != NULL);
	return 0;
}

/*******************************************************************************
 * Internal functions
 ******************************************************************************/
int aml_scratch_seq_doit(struct aml_scratch_seq_data *scratch,
			      struct aml_scratch_request_seq *req)
{
	assert(scratch != NULL);
	assert(req != NULL);
	return aml_dma_async_copy(scratch->dma, &req->dma_req,
				  req->tiling, req->dstptr, req->dstid,
				  req->tiling, req->srcptr, req->srcid);
}

struct aml_scratch_seq_ops aml_scratch_seq_inner_ops = {
	aml_scratch_seq_doit,
};

/*******************************************************************************
 * Public API
 ******************************************************************************/

/* TODO: not thread-safe */

int aml_scratch_seq_create_request(struct aml_scratch_data *d,
				   struct aml_scratch_request **r,
				   int type, va_list ap)
{
	assert(d != NULL);
	assert(r != NULL);
	struct aml_scratch_seq *scratch =
		(struct aml_scratch_seq *)d;

	struct aml_scratch_request_seq *req;

	pthread_mutex_lock(&scratch->data.lock);
	req = aml_vector_add(&scratch->data.requests);
	/* init the request */
	if(type == AML_SCRATCH_REQUEST_TYPE_PUSH)
	{
		int scratchid;
		int *srcid;
		void *srcptr;
		void *scratchptr;

		srcptr = va_arg(ap, void *);
		srcid = va_arg(ap, int *);
		scratchptr = va_arg(ap, void *);
		scratchid = va_arg(ap, int);

		/* find destination tile */
		int *slot = aml_vector_get(&scratch->data.tilemap, scratchid);
		assert(slot != NULL);
		*srcid = *slot;

		/* init request */
		aml_scratch_request_seq_init(req, type, scratch->data.tiling,
					     srcptr, *srcid,
					     scratchptr, scratchid);
	}
	else if(type == AML_SCRATCH_REQUEST_TYPE_PULL)
	{
		int *scratchid;
		int srcid;
		void *srcptr;
		void *scratchptr;

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
		int slot = aml_vector_find(&scratch->data.tilemap, srcid);
		if(slot == -1)
			slot = aml_vector_find(&scratch->data.tilemap, -1);
		assert(slot != -1);
		int *tile = aml_vector_get(&scratch->data.tilemap, slot);
		*tile = srcid;
		*scratchid = slot;

		/* init request */
		aml_scratch_request_seq_init(req, type,
					     scratch->data.tiling,
					     scratchptr, *scratchid,
					     srcptr, srcid);
	}
	pthread_mutex_unlock(&scratch->data.lock);
	scratch->ops.doit(&scratch->data, req);

	*r = (struct aml_scratch_request *)req;
	return 0;
}

int aml_scratch_seq_destroy_request(struct aml_scratch_data *d,
					 struct aml_scratch_request *r)
{
	assert(d != NULL);
	assert(r != NULL);
	struct aml_scratch_seq *scratch =
		(struct aml_scratch_seq *)d;

	struct aml_scratch_request_seq *req =
		(struct aml_scratch_request_seq *)r;
	int *tile;

	aml_dma_cancel(scratch->data.dma, req->dma_req);
	aml_scratch_request_seq_destroy(req);

	/* destroy removes the tile from the scratch */
	pthread_mutex_lock(&scratch->data.lock);
	if(req->type == AML_SCRATCH_REQUEST_TYPE_PUSH)
		tile = aml_vector_get(&scratch->data.tilemap,req->srcid);
	else if(req->type == AML_SCRATCH_REQUEST_TYPE_PULL)
		tile = aml_vector_get(&scratch->data.tilemap,req->dstid);
	aml_vector_remove(&scratch->data.tilemap, tile);
	aml_vector_remove(&scratch->data.requests, req);
	pthread_mutex_unlock(&scratch->data.lock);
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
	int *tile;

	/* wait for completion of the request */
	aml_dma_wait(scratch->data.dma, req->dma_req);

	/* cleanup a completed request. In case of push, free up the tile */
	aml_scratch_request_seq_destroy(req);
	pthread_mutex_lock(&scratch->data.lock);
	if(req->type == AML_SCRATCH_REQUEST_TYPE_PUSH)
	{
		tile = aml_vector_get(&scratch->data.tilemap,req->srcid);
		aml_vector_remove(&scratch->data.tilemap, tile);
	}
	aml_vector_remove(&scratch->data.requests, req);
	pthread_mutex_unlock(&scratch->data.lock);
	return 0;
}

void *aml_scratch_seq_baseptr(const struct aml_scratch_data *d)
{
	assert(d != NULL);
	const struct aml_scratch_seq *scratch = (const struct aml_scratch_seq *)d;
	return scratch->data.sch_ptr;
}

int aml_scratch_seq_release(struct aml_scratch_data *d, int scratchid)
{
	assert(d != NULL);
	struct aml_scratch_seq *scratch = (struct aml_scratch_seq *)d;
	int *tile;

	pthread_mutex_lock(&scratch->data.lock);
	tile = aml_vector_get(&scratch->data.tilemap, scratchid);
	if(tile != NULL)
		aml_vector_remove(&scratch->data.tilemap, tile);
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

int aml_scratch_seq_create(struct aml_scratch **d, ...)
{
	va_list ap;
	struct aml_scratch *ret = NULL;
	intptr_t baseptr, dataptr;
	va_start(ap, d);

	/* alloc */
	baseptr = (intptr_t) calloc(1, AML_SCRATCH_SEQ_ALLOCSIZE);
	dataptr = baseptr + sizeof(struct aml_scratch);

	ret = (struct aml_scratch *)baseptr;
	ret->data = (struct aml_scratch_data *)dataptr;

	aml_scratch_seq_vinit(ret, ap);

	va_end(ap);
	*d = ret;
	return 0;
}
int aml_scratch_seq_vinit(struct aml_scratch *d, va_list ap)
{
	d->ops = &aml_scratch_seq_ops;
	struct aml_scratch_seq *scratch = (struct aml_scratch_seq *)d->data;

	scratch->ops = aml_scratch_seq_inner_ops;

	scratch->data.sch_area = va_arg(ap, struct aml_area *);
	scratch->data.src_area = va_arg(ap, struct aml_area *);
	scratch->data.dma = va_arg(ap, struct aml_dma *);
	scratch->data.tiling = va_arg(ap, struct aml_tiling *);
	size_t nbtiles = va_arg(ap, size_t);
	size_t nbreqs = va_arg(ap, size_t);

	/* allocate request array */
	aml_vector_init(&scratch->data.requests, nbreqs,
			sizeof(struct aml_scratch_request_seq),
			offsetof(struct aml_scratch_request_seq, type),
			AML_SCRATCH_REQUEST_TYPE_INVALID);

	/* scratch init */
	aml_vector_init(&scratch->data.tilemap, nbtiles, sizeof(int), 0, -1);
	size_t tilesize = aml_tiling_tilesize(scratch->data.tiling, 0);
	scratch->data.sch_ptr = aml_area_calloc(scratch->data.sch_area,
						nbtiles, tilesize);
	pthread_mutex_init(&scratch->data.lock, NULL);
	return 0;
}
int aml_scratch_seq_init(struct aml_scratch *d, ...)
{
	int err;
	va_list ap;
	va_start(ap, d);
	err = aml_scratch_seq_vinit(d, ap);
	va_end(ap);
	return err;
}

int aml_scratch_seq_destroy(struct aml_scratch *d)
{
	struct aml_scratch_seq *scratch = (struct aml_scratch_seq *)d->data;
	aml_vector_destroy(&scratch->data.requests);
	aml_vector_destroy(&scratch->data.tilemap);
	aml_area_free(scratch->data.sch_area, scratch->data.sch_ptr);
	pthread_mutex_destroy(&scratch->data.lock);
	return 0;
}
