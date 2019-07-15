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
#include "aml/dma/linux-seq.h"

#include <assert.h>
#include <errno.h>
#include <sys/mman.h>

/*******************************************************************************
 * Linux-backed, sequential dma
 * The dma itself is organized into several different components
 * - request types: copy or move
 * - implementation of the request
 * - user API (i.e. generic request creation and call)
 * - how to init the dma
 ******************************************************************************/

/*******************************************************************************
 * Requests:
 ******************************************************************************/

int aml_dma_request_linux_seq_copy_init(struct aml_dma_request_linux_seq *req,
					const struct aml_tiling *dt,
					void *dptr, int dtid,
					const struct aml_tiling *st,
					const void *sptr, int stid)
{
	assert(req != NULL);

	req->type = AML_DMA_REQUEST_TYPE_COPY;
	/* figure out pointers */
	req->dest = aml_tiling_tilestart(dt, dptr, dtid);
	req->src = aml_tiling_tilestart(st, sptr, stid);
	req->size = aml_tiling_tilesize(st, stid);
	/* TODO: assert size match */
	return 0;
}

int aml_dma_request_linux_seq_copy_destroy(struct aml_dma_request_linux_seq *r)
{
	assert(r != NULL);
	return 0;
}

/*******************************************************************************
 * Internal functions
 ******************************************************************************/
int aml_dma_linux_seq_do_copy(struct aml_dma_linux_seq_data *dma,
			      struct aml_dma_request_linux_seq *req)
{
	assert(dma != NULL);
	assert(req != NULL);
	memcpy(req->dest, req->src, req->size);
	return 0;
}

struct aml_dma_linux_seq_ops aml_dma_linux_seq_inner_ops = {
	aml_dma_linux_seq_do_copy,
};

/*******************************************************************************
 * Public API
 ******************************************************************************/

int aml_dma_linux_seq_create_request(struct aml_dma_data *d,
				     struct aml_dma_request **r,
				     int type, va_list ap)
{
	assert(d != NULL);
	assert(r != NULL);
	struct aml_dma_linux_seq *dma =
		(struct aml_dma_linux_seq *)d;

	struct aml_dma_request_linux_seq *req;

	pthread_mutex_lock(&dma->data.lock);
	req = aml_vector_add(dma->data.requests);

	/* init the request */
	if (type == AML_DMA_REQUEST_TYPE_COPY) {
		struct aml_tiling *dt, *st;
		void *dptr, *sptr;
		int dtid, stid;

		dt = va_arg(ap, struct aml_tiling *);
		dptr = va_arg(ap, void *);
		dtid = va_arg(ap, int);
		st = va_arg(ap, struct aml_tiling *);
		sptr = va_arg(ap, void *);
		stid = va_arg(ap, int);
		aml_dma_request_linux_seq_copy_init(req, dt, dptr, dtid,
						    st, sptr, stid);
	}
	pthread_mutex_unlock(&dma->data.lock);
	*r = (struct aml_dma_request *)req;
	return 0;
}

int aml_dma_linux_seq_destroy_request(struct aml_dma_data *d,
				      struct aml_dma_request *r)
{
	assert(d != NULL);
	assert(r != NULL);
	struct aml_dma_linux_seq *dma =
		(struct aml_dma_linux_seq *)d;

	struct aml_dma_request_linux_seq *req =
		(struct aml_dma_request_linux_seq *)r;

	if (req->type == AML_DMA_REQUEST_TYPE_COPY)
		aml_dma_request_linux_seq_copy_destroy(req);

	/* enough to remove from request vector */
	pthread_mutex_lock(&dma->data.lock);
	aml_vector_remove(dma->data.requests, req);
	pthread_mutex_unlock(&dma->data.lock);
	return 0;
}

int aml_dma_linux_seq_wait_request(struct aml_dma_data *d,
				   struct aml_dma_request *r)
{
	assert(d != NULL);
	assert(r != NULL);
	struct aml_dma_linux_seq *dma = (struct aml_dma_linux_seq *)d;
	struct aml_dma_request_linux_seq *req =
		(struct aml_dma_request_linux_seq *)r;

	/* execute */
	if (req->type == AML_DMA_REQUEST_TYPE_COPY)
		dma->ops.do_copy(&dma->data, req);

	/* destroy a completed request */
	aml_dma_linux_seq_destroy_request(d, r);
	return 0;
}

struct aml_dma_ops aml_dma_linux_seq_ops = {
	aml_dma_linux_seq_create_request,
	aml_dma_linux_seq_destroy_request,
	aml_dma_linux_seq_wait_request,
};

/*******************************************************************************
 * Init functions:
 ******************************************************************************/

int aml_dma_linux_seq_create(struct aml_dma **dma, size_t nbreqs)
{
	struct aml_dma *ret = NULL;
	struct aml_dma_linux_seq *d;

	if (dma == NULL)
		return -AML_EINVAL;

	*dma = NULL;

	/* alloc */
	ret = calloc(1, sizeof(struct aml_dma));
	if (ret == NULL)
		return -AML_ENOMEM;

	ret->data = calloc(1, sizeof(struct aml_dma_linux_seq));
	if (ret->data == NULL) {
		free(ret);
		return -AML_ENOMEM;
	}
	ret->ops = &aml_dma_linux_seq_ops;
	d = (struct aml_dma_linux_seq *)ret->data;

	d->ops = aml_dma_linux_seq_inner_ops;
	aml_vector_create(&d->data.requests, nbreqs,
			  sizeof(struct aml_dma_request_linux_seq),
			  offsetof(struct aml_dma_request_linux_seq, type),
			  AML_DMA_REQUEST_TYPE_INVALID);
	pthread_mutex_init(&d->data.lock, NULL);

	*dma = ret;
	return 0;
}

void aml_dma_linux_seq_destroy(struct aml_dma **dma)
{
	struct aml_dma *d;
	struct aml_dma_linux_seq *l;

	if (dma == NULL)
		return;
	d = *dma;
	if (d == NULL || d->data == NULL)
		return;
	l = (struct aml_dma_linux_seq *)d->data;
	aml_vector_destroy(&l->data.requests);
	pthread_mutex_destroy(&l->data.lock);
	free(l);
	free(d);
	*dma = NULL;
}
