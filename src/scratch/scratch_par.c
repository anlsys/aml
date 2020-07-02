
/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *******************************************************************************/

#include <assert.h>

#include "aml.h"

#include "aml/layout/dense.h"
#include "aml/scratch/par.h"

/*******************************************************************************
 * Parallel scratchpad
 * The scratch itself is organized into several different components
 * - request types: push and pull
 * - implementation of the request
 * - user API (i.e. generic request creation and call)
 * - how to init the scratch
 ******************************************************************************/

/*******************************************************************************
 * Requests:
 ******************************************************************************/

int aml_scratch_request_par_init(struct aml_scratch_request_par *req,
                                 int type,
                                 struct aml_layout *dest,
                                 int destid,
                                 struct aml_layout *src,
                                 int srcid,
                                 struct aml_scratch_par *scratch)

{
	assert(req != NULL);
	req->type = type;
	req->scratch = scratch;
	req->src = src;
	req->srcid = srcid;
	req->dst = dest;
	req->dstid = destid;
	return 0;
}

int aml_scratch_request_par_destroy(struct aml_scratch_request_par *r)
{
	assert(r != NULL);
	return 0;
}

/*******************************************************************************
 * Internal functions
 ******************************************************************************/
void *aml_scratch_par_do_thread(void *arg)
{
	struct aml_scratch_request_par *req =
	        (struct aml_scratch_request_par *)arg;
	struct aml_scratch_par *scratch = req->scratch;

	pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, NULL);
	aml_dma_copy(scratch->data.dma, req->dst, req->src, NULL, NULL);
	return NULL;
}

struct aml_scratch_par_inner_ops aml_scratch_par_inner_ops = {
        aml_scratch_par_do_thread,
};

/*******************************************************************************
 * Public API
 ******************************************************************************/

int aml_scratch_par_create_request(struct aml_scratch_data *d,
                                   struct aml_scratch_request **r,
                                   int type,
                                   struct aml_layout **dest,
                                   int *destid,
                                   struct aml_layout *src,
                                   int srcid)
{
	assert(d != NULL);
	assert(r != NULL);
	struct aml_scratch_par *scratch = (struct aml_scratch_par *)d;

	struct aml_scratch_request_par *req;

	pthread_mutex_lock(&scratch->data.lock);
	req = aml_vector_add(scratch->data.requests);
	/* init the request */
	if (type == AML_SCRATCH_REQUEST_TYPE_PUSH) {

		/* find destination tile */
		int *slot = aml_vector_get(scratch->data.tilemap, srcid);

		assert(slot != NULL);
		*destid = *slot;
		*dest = aml_tiling_index_byid(scratch->data.src_tiling,
		                              *destid);

		/* init request */
		aml_scratch_request_par_init(req, type, *dest, *destid, src,
		                             srcid, scratch);
	} else if (type == AML_SCRATCH_REQUEST_TYPE_PULL) {
		int slot, *tile;

		/* find destination tile
		 * We don't use add here because adding a tile means allocating
		 * new tiles on the sch_area too. */
		slot = aml_vector_find(scratch->data.tilemap, srcid);
		if (slot == -1) {
			slot = aml_vector_find(scratch->data.tilemap, -1);
			assert(slot != -1);
			tile = aml_vector_get(scratch->data.tilemap, slot);
			*tile = srcid;
		} else
			type = AML_SCRATCH_REQUEST_TYPE_NOOP;

		/* save the key */
		*destid = slot;
		*dest = aml_tiling_index_byid(scratch->data.scratch_tiling,
		                              slot);

		/* init request */
		aml_scratch_request_par_init(req, type, *dest, *destid, src,
		                             srcid, scratch);
	}
	pthread_mutex_unlock(&scratch->data.lock);
	/* thread creation */
	if (req->type != AML_SCRATCH_REQUEST_TYPE_NOOP)
		pthread_create(&req->thread, NULL, scratch->ops.do_thread, req);
	*r = (struct aml_scratch_request *)req;
	return 0;
}

int aml_scratch_par_destroy_request(struct aml_scratch_data *d,
                                    struct aml_scratch_request *r)
{
	assert(d != NULL);
	assert(r != NULL);
	struct aml_scratch_par *scratch = (struct aml_scratch_par *)d;

	struct aml_scratch_request_par *req =
	        (struct aml_scratch_request_par *)r;
	int *tile = NULL;

	if (req->type != AML_SCRATCH_REQUEST_TYPE_NOOP) {
		pthread_cancel(req->thread);
		pthread_join(req->thread, NULL);
	}

	aml_scratch_request_par_destroy(req);

	/* destroy removes the tile from the scratch */
	pthread_mutex_lock(&scratch->data.lock);
	if (req->type == AML_SCRATCH_REQUEST_TYPE_PUSH)
		tile = aml_vector_get(scratch->data.tilemap, req->srcid);
	else if (req->type == AML_SCRATCH_REQUEST_TYPE_PULL)
		tile = aml_vector_get(scratch->data.tilemap, req->dstid);
	if (tile != NULL)
		aml_vector_remove(scratch->data.tilemap, tile);
	aml_vector_remove(scratch->data.requests, req);
	pthread_mutex_unlock(&scratch->data.lock);
	return 0;
}

int aml_scratch_par_wait_request(struct aml_scratch_data *d,
                                 struct aml_scratch_request *r)
{
	assert(d != NULL);
	assert(r != NULL);
	struct aml_scratch_par *scratch = (struct aml_scratch_par *)d;
	struct aml_scratch_request_par *req =
	        (struct aml_scratch_request_par *)r;
	int *tile;

	/* wait for completion of the request */
	if (req->type != AML_SCRATCH_REQUEST_TYPE_NOOP)
		pthread_join(req->thread, NULL);

	/* cleanup a completed request. In case of push, free up the tile */
	aml_scratch_request_par_destroy(req);
	pthread_mutex_lock(&scratch->data.lock);
	if (req->type == AML_SCRATCH_REQUEST_TYPE_PUSH) {
		tile = aml_vector_get(scratch->data.tilemap, req->srcid);
		aml_vector_remove(scratch->data.tilemap, tile);
	}
	aml_vector_remove(scratch->data.requests, req);
	pthread_mutex_unlock(&scratch->data.lock);
	return 0;
}

int aml_scratch_par_release(struct aml_scratch_data *d, int scratchid)
{
	assert(d != NULL);
	struct aml_scratch_par *scratch = (struct aml_scratch_par *)d;
	int *tile;

	pthread_mutex_lock(&scratch->data.lock);
	tile = aml_vector_get(scratch->data.tilemap, scratchid);
	if (tile != NULL)
		aml_vector_remove(scratch->data.tilemap, tile);
	pthread_mutex_unlock(&scratch->data.lock);
	return 0;
}

struct aml_scratch_ops aml_scratch_par_ops = {
        aml_scratch_par_create_request,
        aml_scratch_par_destroy_request,
        aml_scratch_par_wait_request,
        aml_scratch_par_release,
};

/*******************************************************************************
 * Init functions:
 ******************************************************************************/

int aml_scratch_par_create(struct aml_scratch **scratch,
                           struct aml_dma *dma,
                           struct aml_tiling *src_tiling,
                           struct aml_tiling *scratch_tiling,
                           size_t nbreqs)
{
	struct aml_scratch *ret = NULL;
	struct aml_scratch_par *s;

	if (scratch == NULL || dma == NULL || src_tiling == NULL ||
	    scratch_tiling == NULL)
		return -AML_EINVAL;

	*scratch = NULL;

	ret = AML_INNER_MALLOC(struct aml_scratch, struct aml_scratch_par);
	if (ret == NULL)
		return -AML_ENOMEM;

	ret->ops = &aml_scratch_par_ops;
	ret->data = AML_INNER_MALLOC_GET_FIELD(ret, 2, struct aml_scratch,
	                                       struct aml_scratch_par);
	s = (struct aml_scratch_par *)ret->data;
	s->ops = aml_scratch_par_inner_ops;

	s->data.dma = dma;
	s->data.src_tiling = src_tiling;
	s->data.scratch_tiling = scratch_tiling;

	/* allocate request array */
	aml_vector_create(&s->data.requests, nbreqs,
	                  sizeof(struct aml_scratch_request_par),
	                  offsetof(struct aml_scratch_request_par, type),
	                  AML_SCRATCH_REQUEST_TYPE_INVALID);

	/* s init */
	size_t nbtiles = aml_tiling_ntiles(scratch_tiling);

	aml_vector_create(&s->data.tilemap, nbtiles, sizeof(int), 0, -1);
	pthread_mutex_init(&s->data.lock, NULL);

	*scratch = ret;
	return 0;
}

void aml_scratch_par_destroy(struct aml_scratch **scratch)
{
	struct aml_scratch *s;
	struct aml_scratch_par *inner;

	if (scratch == NULL)
		return;
	s = *scratch;
	if (s == NULL)
		return;

	assert(s->data != NULL);
	inner = (struct aml_scratch_par *)s->data;
	aml_vector_destroy(&inner->data.requests);
	aml_vector_destroy(&inner->data.tilemap);
	pthread_mutex_destroy(&inner->data.lock);
	free(s);
	*scratch = NULL;
}
