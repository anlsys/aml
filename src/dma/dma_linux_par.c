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
#include "aml/dma/linux-par.h"

#include <assert.h>
#include <errno.h>
#include <sys/mman.h>

/*******************************************************************************
 * Linux-backed, paruential dma
 * The dma itself is organized into several different components
 * - request types: copy
 * - implementation of the request
 * - user API (i.e. generic request creation and call)
 * - how to init the dma
 ******************************************************************************/

/*******************************************************************************
 * Requests:
 ******************************************************************************/

int aml_dma_request_linux_par_copy_init(struct aml_dma_request_linux_par *req,
					struct aml_tiling *dt,
					void *dptr, int dtid,
					struct aml_tiling *st,
					void *sptr, int stid)
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

int aml_dma_request_linux_par_copy_destroy(struct aml_dma_request_linux_par *r)
{
	assert(r != NULL);
	return 0;
}

/*******************************************************************************
 * Internal functions
 ******************************************************************************/

void *aml_dma_linux_par_do_thread(void *arg)
{
	struct aml_dma_linux_par_thread_data *data =
		(struct aml_dma_linux_par_thread_data *)arg;

	if (data->req->type == AML_DMA_REQUEST_TYPE_COPY)
		data->dma->ops.do_copy(&data->dma->data, data->req, data->tid);
	return NULL;
}

int aml_dma_linux_par_do_copy(struct aml_dma_linux_par_data *dma,
			      struct aml_dma_request_linux_par *req, size_t tid)
{
	assert(dma != NULL);
	assert(req != NULL);

	/* chunk memory */
	size_t nbthreads = dma->nbthreads;
	size_t chunksize = req->size / nbthreads;

	void *dest = (void *)((intptr_t)req->dest + tid * chunksize);
	void *src = (void *)((intptr_t)req->src + tid * chunksize);

	if (tid == nbthreads - 1 && req->size > chunksize * nbthreads)
		chunksize += req->size % nbthreads;

	memcpy(dest, src, chunksize);
	return 0;
}

struct aml_dma_linux_par_ops aml_dma_linux_par_inner_ops = {
	aml_dma_linux_par_do_thread,
	aml_dma_linux_par_do_copy,
};

/*******************************************************************************
 * Public API
 ******************************************************************************/

int aml_dma_linux_par_create_request(struct aml_dma_data *d,
				     struct aml_dma_request **r,
				     int type, va_list ap)
{
	assert(d != NULL);
	assert(r != NULL);
	struct aml_dma_linux_par *dma =
		(struct aml_dma_linux_par *)d;

	struct aml_dma_request_linux_par *req;

	pthread_mutex_lock(&dma->data.lock);
	req = aml_vector_add(&dma->data.requests);

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
		aml_dma_request_linux_par_copy_init(req, dt, dptr, dtid,
						    st, sptr, stid);
	}
	pthread_mutex_unlock(&dma->data.lock);

	for (size_t i = 0; i < dma->data.nbthreads; i++) {
		struct aml_dma_linux_par_thread_data *rd = &req->thread_data[i];

		rd->req = req;
		rd->dma = dma;
		rd->tid = i;
		pthread_create(&rd->thread, NULL, dma->ops.do_thread, rd);
	}
	*r = (struct aml_dma_request *)req;
	return 0;
}

int aml_dma_linux_par_destroy_request(struct aml_dma_data *d,
				      struct aml_dma_request *r)
{
	assert(d != NULL);
	assert(r != NULL);
	struct aml_dma_linux_par *dma =
		(struct aml_dma_linux_par *)d;

	struct aml_dma_request_linux_par *req =
		(struct aml_dma_request_linux_par *)r;

	/* we cancel and join, instead of killing, for a cleaner result */
	for (size_t i = 0; i < dma->data.nbthreads; i++) {
		pthread_cancel(req->thread_data[i].thread);
		pthread_join(req->thread_data[i].thread, NULL);
	}

	if (req->type == AML_DMA_REQUEST_TYPE_COPY)
		aml_dma_request_linux_par_copy_destroy(req);

	pthread_mutex_lock(&dma->data.lock);
	aml_vector_remove(&dma->data.requests, req);
	pthread_mutex_unlock(&dma->data.lock);
	return 0;
}

int aml_dma_linux_par_wait_request(struct aml_dma_data *d,
				   struct aml_dma_request *r)
{
	assert(d != NULL);
	assert(r != NULL);
	struct aml_dma_linux_par *dma = (struct aml_dma_linux_par *)d;
	struct aml_dma_request_linux_par *req =
		(struct aml_dma_request_linux_par *)r;

	for (size_t i = 0; i < dma->data.nbthreads; i++)
		pthread_join(req->thread_data[i].thread, NULL);

	/* destroy a completed request */
	if (req->type == AML_DMA_REQUEST_TYPE_COPY)
		aml_dma_request_linux_par_copy_destroy(req);

	pthread_mutex_lock(&dma->data.lock);
	aml_vector_remove(&dma->data.requests, req);
	pthread_mutex_unlock(&dma->data.lock);
	return 0;
}

struct aml_dma_ops aml_dma_linux_par_ops = {
	aml_dma_linux_par_create_request,
	aml_dma_linux_par_destroy_request,
	aml_dma_linux_par_wait_request,
};

/*******************************************************************************
 * Init functions:
 ******************************************************************************/

int aml_dma_linux_par_create(struct aml_dma **d, size_t nbreqs,
			     size_t nbthreads)
{
	struct aml_dma *ret = NULL;
	intptr_t baseptr, dataptr;
	int err;

	if (d == NULL)
		return -AML_EINVAL;

	/* alloc */
	baseptr = (intptr_t) calloc(1, AML_DMA_LINUX_PAR_ALLOCSIZE);
	if (baseptr == 0) {
		*d = NULL;
		return -AML_ENOMEM;
	}
	dataptr = baseptr + sizeof(struct aml_dma);

	ret = (struct aml_dma *)baseptr;
	ret->data = (struct aml_dma_data *)dataptr;
	ret->ops = &aml_dma_linux_par_ops;

	err = aml_dma_linux_par_init(ret, nbreqs, nbthreads);
	if (err) {
		*d = NULL;
		free(ret);
		return err;
	}

	*d = ret;
	return 0;
}

int aml_dma_linux_par_init(struct aml_dma *d, size_t nbreqs,
			   size_t nbthreads)
{
	struct aml_dma_linux_par *dma;

	if (d == NULL || d->data == NULL)
		return -AML_EINVAL;
	dma = (struct aml_dma_linux_par *)d->data;
	dma->ops = aml_dma_linux_par_inner_ops;

	/* allocate request array */
	dma->data.nbthreads = nbthreads;
	aml_vector_init(&dma->data.requests, nbreqs,
			sizeof(struct aml_dma_request_linux_par),
			offsetof(struct aml_dma_request_linux_par, type),
			AML_DMA_REQUEST_TYPE_INVALID);
	for (size_t i = 0; i < nbreqs; i++) {
		struct aml_dma_request_linux_par *req =
			aml_vector_get(&dma->data.requests, i);

		req->thread_data = calloc(dma->data.nbthreads,
				sizeof(struct aml_dma_linux_par_thread_data));
	}
	pthread_mutex_init(&dma->data.lock, NULL);
	return 0;
}

void aml_dma_linux_par_fini(struct aml_dma *d)
{
	struct aml_dma_linux_par *dma;

	if (d == NULL || d->data == NULL)
		return;
	dma = (struct aml_dma_linux_par *)d->data;
	for (size_t i = 0; i < aml_vector_size(&dma->data.requests); i++) {
		struct aml_dma_request_linux_par *req =
			aml_vector_get(&dma->data.requests, i);

		free(req->thread_data);
	}
	aml_vector_fini(&dma->data.requests);
	pthread_mutex_destroy(&dma->data.lock);
}

void aml_dma_linux_par_destroy(struct aml_dma **d)
{
	if (d == NULL)
		return;
	aml_dma_linux_par_fini(*d);
	free(*d);
	*d = NULL;
}
