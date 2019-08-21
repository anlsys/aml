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
#include "aml/layout/dense.h"

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

int aml_dma_request_linux_seq_create(struct aml_dma_request_linux_seq **req,
				     int uuid)
{
	assert(req != NULL);
	*req = calloc(1, sizeof(struct aml_dma_request_linux_seq));
	if (*req == NULL)
		return -AML_ENOMEM;
	(*req)->uuid = uuid;
	return 0;
}

void aml_dma_request_linux_seq_destroy(struct aml_dma_request_linux_seq **req)
{
	assert(req != NULL);
	free(*req);
	*req = NULL;
}

void aml_dma_linux_seq_request_data_init(
				struct aml_dma_linux_seq_request_data *req,
				int type,
				struct aml_layout *dest,
				struct aml_layout *src)
{
	assert(req != NULL);
	req->type = type;
	req->dest = dest;
	req->src = src;
}

/*******************************************************************************
 * Internal functions
 ******************************************************************************/

int aml_dma_linux_seq_do_copy(struct aml_dma_linux_seq_data *dma,
			      struct aml_dma_linux_seq_request_data *req)
{
	assert(dma != NULL);
	assert(req != NULL);
	aml_copy_layout_generic(req->dest, req->src);
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
	struct aml_dma_request_linux_seq *ret;
	struct aml_dma_linux_seq_request_data *req;
	int err = AML_SUCCESS;

	pthread_mutex_lock(&dma->data.lock);
	req = aml_vector_add(dma->data.requests);

	/* init the request */
	if (type == AML_DMA_REQUEST_TYPE_LAYOUT) {
		struct aml_layout *dl, *sl;

		dl = va_arg(ap, struct aml_layout *);
		sl = va_arg(ap, struct aml_layout *);
		if (dl == NULL || sl == NULL) {
			err = -AML_EINVAL;
			goto unlock;
		}
		aml_dma_linux_seq_request_data_init(req,
						    AML_DMA_REQUEST_TYPE_LAYOUT,
						    dl, sl);
	} else if (type == AML_DMA_REQUEST_TYPE_PTR) {
		struct aml_layout *dl, *sl;
		void *dp, *sp;
		size_t sz;

		dp = va_arg(ap, void *);
		sp = va_arg(ap, void *);
		sz = va_arg(ap, size_t);
		if (dp == NULL || sp == NULL || sz == 0) {
			err = -AML_EINVAL;
			goto unlock;
		}
		/* simple 1D layout, none of the parameters really matter, as
		 * long as the copy generates a single memcpy.
		 */
		aml_layout_dense_create(&dl, dp, 0, 1, 1, &sz, NULL, NULL);
		aml_layout_dense_create(&sl, sp, 0, 1, 1, &sz, NULL, NULL);
		aml_dma_linux_seq_request_data_init(req,
						    AML_DMA_REQUEST_TYPE_PTR,
						    dl, sl);
	} else
		err = -AML_EINVAL;
unlock:
	pthread_mutex_unlock(&dma->data.lock);
	if (req->type != AML_DMA_REQUEST_TYPE_INVALID) {
		int uuid = aml_vector_getid(dma->data.requests, req);

		assert(uuid != AML_DMA_REQUEST_TYPE_INVALID);
		aml_dma_request_linux_seq_create(&ret, uuid);
		*r = (struct aml_dma_request *)ret;
	}
	return err;
}

int aml_dma_linux_seq_destroy_request(struct aml_dma_data *d,
				      struct aml_dma_request **r)
{
	assert(d != NULL);
	assert(r != NULL);
	struct aml_dma_linux_seq *dma =
		(struct aml_dma_linux_seq *)d;
	struct aml_dma_request_linux_seq *req;
	struct aml_dma_linux_seq_request_data *inner_req;

	if (*r == NULL)
		return -AML_EINVAL;
	req = (struct aml_dma_request_linux_seq *)*r;

	inner_req = aml_vector_get(dma->data.requests, req->uuid);
	if (inner_req == NULL)
		return -AML_EINVAL;

	pthread_mutex_lock(&dma->data.lock);
	if (inner_req->type == AML_DMA_REQUEST_TYPE_PTR) {
		aml_layout_dense_destroy(&inner_req->dest);
		aml_layout_dense_destroy(&inner_req->src);
	}

	aml_vector_remove(dma->data.requests, inner_req);
	pthread_mutex_unlock(&dma->data.lock);
	aml_dma_request_linux_seq_destroy(&req);
	*r = NULL;
	return 0;
}

int aml_dma_linux_seq_wait_request(struct aml_dma_data *d,
				   struct aml_dma_request **r)
{
	assert(d != NULL);
	assert(r != NULL);
	struct aml_dma_linux_seq *dma = (struct aml_dma_linux_seq *)d;
	struct aml_dma_request_linux_seq *req;
	struct aml_dma_linux_seq_request_data *inner_req;

	if (*r == NULL)
		return -AML_EINVAL;
	req = (struct aml_dma_request_linux_seq *)*r;

	inner_req = aml_vector_get(dma->data.requests, req->uuid);
	if (inner_req == NULL)
		return -AML_EINVAL;

	/* execute */
	if (inner_req->type != AML_DMA_REQUEST_TYPE_INVALID)
		dma->ops.do_copy(&dma->data, inner_req);

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

	ret = AML_INNER_MALLOC_2(struct aml_dma, struct aml_dma_linux_seq);
	if (ret == NULL)
		return -AML_ENOMEM;

	ret->data = AML_INNER_MALLOC_NEXTPTR(ret, struct aml_dma,
					     struct aml_dma_linux_seq);
	ret->ops = &aml_dma_linux_seq_ops;
	d = (struct aml_dma_linux_seq *)ret->data;

	d->ops = aml_dma_linux_seq_inner_ops;
	aml_vector_create(&d->data.requests, nbreqs,
			  sizeof(struct aml_dma_linux_seq_request_data),
			  offsetof(struct aml_dma_linux_seq_request_data, type),
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
	if (d == NULL)
		return;

	assert(d->data != NULL);
	l = (struct aml_dma_linux_seq *)d->data;
	aml_vector_destroy(&l->data.requests);
	pthread_mutex_destroy(&l->data.lock);
	free(d);
	*dma = NULL;
}
