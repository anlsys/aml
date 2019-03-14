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
#include <errno.h>
#include <sys/mman.h>

/*******************************************************************************
 * Linux-backed, sequential dma
 * The dma itself is organized into several different components
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
				     va_list ap)
{
	assert(d != NULL);
	assert(r != NULL);
	struct aml_dma_linux_seq *dma =
		(struct aml_dma_linux_seq *)d;

	struct aml_dma_request_linux_seq *req;

	pthread_mutex_lock(&dma->data.lock);
	req = aml_vector_add(&dma->data.requests);

	/* init the request */
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

	aml_dma_request_linux_seq_copy_destroy(req);

	/* enough to remove from request vector */
	pthread_mutex_lock(&dma->data.lock);
	aml_vector_remove(&dma->data.requests, req);
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

int aml_dma_linux_seq_create(struct aml_dma **d, ...)
{
	va_list ap;
	struct aml_dma *ret = NULL;
	intptr_t baseptr, dataptr;
	va_start(ap, d);

	/* alloc */
	baseptr = (intptr_t) calloc(1, AML_DMA_LINUX_SEQ_ALLOCSIZE);
	dataptr = baseptr + sizeof(struct aml_dma);

	ret = (struct aml_dma *)baseptr;
	ret->data = (struct aml_dma_data *)dataptr;

	aml_dma_linux_seq_vinit(ret, ap);

	va_end(ap);
	*d = ret;
	return 0;
}
int aml_dma_linux_seq_vinit(struct aml_dma *d, va_list ap)
{
	d->ops = &aml_dma_linux_seq_ops;
	struct aml_dma_linux_seq *dma = (struct aml_dma_linux_seq *)d->data;

	dma->ops = aml_dma_linux_seq_inner_ops;

	/* request vector */
	size_t nbreqs = va_arg(ap, size_t);
	aml_vector_init(&dma->data.requests, nbreqs,
			sizeof(struct aml_dma_request_linux_seq),
			sizeof(int), 0);
	pthread_mutex_init(&dma->data.lock, NULL);
	return 0;
}
int aml_dma_linux_seq_init(struct aml_dma *d, ...)
{
	int err;
	va_list ap;
	va_start(ap, d);
	err = aml_dma_linux_seq_vinit(d, ap);
	va_end(ap);
	return err;
}

int aml_dma_linux_seq_destroy(struct aml_dma *d)
{
	struct aml_dma_linux_seq *dma = (struct aml_dma_linux_seq *)d->data;
	aml_vector_destroy(&dma->data.requests);
	pthread_mutex_destroy(&dma->data.lock);
	return 0;
}
