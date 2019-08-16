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
#include "aml/layout/dense.h"

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
int aml_dma_request_linux_par_create(struct aml_dma_request_linux_par **req,
				     int uuid)
{
	assert(req != NULL);
	*req = calloc(1, sizeof(struct aml_dma_request_linux_par));
	if (*req == NULL)
		return -AML_ENOMEM;
	(*req)->uuid = uuid;
	return 0;
}

void aml_dma_request_linux_par_destroy(struct aml_dma_request_linux_par **req)
{
	assert(req != NULL);
	free(*req);
	*req = NULL;
}

int aml_dma_linux_par_request_data_init(
				struct aml_dma_linux_par_request_data *req,
				int type,
				struct aml_layout *dest,
				struct aml_layout *src)
{
	assert(req != NULL);
	req->type = type;
	req->dest = dest;
	req->src = src;
	return 0;
}

/*******************************************************************************
 * Internal functions
 ******************************************************************************/

void *aml_dma_linux_par_do_thread(void *arg)
{
	struct aml_dma_linux_par_request_data *req =
		(struct aml_dma_linux_par_request_data *)arg;

	pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, NULL);
	if (req->type != AML_DMA_REQUEST_TYPE_INVALID)
		aml_copy_layout_generic(req->dest, req->src);
	return NULL;
}

struct aml_dma_linux_par_ops aml_dma_linux_par_inner_ops = {
	aml_dma_linux_par_do_thread,
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
	struct aml_dma_request_linux_par *ret;
	struct aml_dma_linux_par_request_data *req;
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
		aml_dma_linux_par_request_data_init(req,
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
		aml_dma_linux_par_request_data_init(req,
						    AML_DMA_REQUEST_TYPE_PTR,
						    dl, sl);
	} else
		err = -AML_EINVAL;
unlock:
	pthread_mutex_unlock(&dma->data.lock);
	if (req->type != AML_DMA_REQUEST_TYPE_INVALID) {
		int uuid = aml_vector_getid(dma->data.requests, req);

		pthread_create(&req->thread, NULL, dma->ops.do_thread, req);
		aml_dma_request_linux_par_create(&ret, uuid);
		*r = (struct aml_dma_request *)ret;
	}
	return err;
}

int aml_dma_linux_par_destroy_request(struct aml_dma_data *d,
				      struct aml_dma_request **r)
{
	assert(d != NULL);
	assert(r != NULL);
	struct aml_dma_linux_par *dma =
		(struct aml_dma_linux_par *)d;
	struct aml_dma_request_linux_par *req;
	struct aml_dma_linux_par_request_data *inner_req;

	if (*r == NULL)
		return -AML_EINVAL;
	req = (struct aml_dma_request_linux_par *)*r;

	inner_req = aml_vector_get(dma->data.requests, req->uuid);
	if (inner_req == NULL)
		return -AML_EINVAL;

	/* we cancel and join, instead of killing, for a cleaner result */
	if (inner_req->type != AML_DMA_REQUEST_TYPE_INVALID) {
		pthread_cancel(inner_req->thread);
		pthread_join(inner_req->thread, NULL);
	}

	if (inner_req->type == AML_DMA_REQUEST_TYPE_PTR) {
		aml_layout_dense_destroy(&inner_req->dest);
		aml_layout_dense_destroy(&inner_req->src);
	}
	pthread_mutex_lock(&dma->data.lock);
	aml_vector_remove(dma->data.requests, inner_req);
	pthread_mutex_unlock(&dma->data.lock);
	aml_dma_request_linux_par_destroy(&req);
	*r = NULL;
	return 0;
}

int aml_dma_linux_par_wait_request(struct aml_dma_data *d,
				   struct aml_dma_request **r)
{
	assert(d != NULL);
	assert(r != NULL);
	struct aml_dma_linux_par *dma = (struct aml_dma_linux_par *)d;
	struct aml_dma_request_linux_par *req;
	struct aml_dma_linux_par_request_data *inner_req;

	if (*r == NULL)
		return -AML_EINVAL;
	req = (struct aml_dma_request_linux_par *)*r;

	inner_req = aml_vector_get(dma->data.requests, req->uuid);
	if (inner_req == NULL)
		return -AML_EINVAL;

	if (inner_req->type != AML_DMA_REQUEST_TYPE_INVALID)
		pthread_join(inner_req->thread, NULL);

	if (inner_req->type == AML_DMA_REQUEST_TYPE_PTR) {
		aml_layout_dense_destroy(&inner_req->dest);
		aml_layout_dense_destroy(&inner_req->src);
	}
	pthread_mutex_lock(&dma->data.lock);
	aml_vector_remove(dma->data.requests, inner_req);
	pthread_mutex_unlock(&dma->data.lock);
	aml_dma_request_linux_par_destroy(&req);
	*r = NULL;
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

int aml_dma_linux_par_create(struct aml_dma **dma, size_t nbreqs)
{
	struct aml_dma *ret = NULL;
	struct aml_dma_linux_par *d;

	if (dma == NULL)
		return -AML_EINVAL;

	*dma = NULL;

	ret = AML_INNER_MALLOC_2(struct aml_dma, struct aml_dma_linux_par);
	if (ret == NULL)
		return -AML_ENOMEM;

	ret->data = AML_INNER_MALLOC_NEXTPTR(ret, struct aml_dma,
					     struct aml_dma_linux_par);
	ret->ops = &aml_dma_linux_par_ops;
	d = (struct aml_dma_linux_par *)ret->data;
	d->ops = aml_dma_linux_par_inner_ops;

	/* allocate request array */
	aml_vector_create(&d->data.requests, nbreqs,
			  sizeof(struct aml_dma_linux_par_request_data),
			  offsetof(struct aml_dma_linux_par_request_data, type),
			  AML_DMA_REQUEST_TYPE_INVALID);
	pthread_mutex_init(&d->data.lock, NULL);

	*dma = ret;
	return 0;
}

void aml_dma_linux_par_destroy(struct aml_dma **d)
{
	struct aml_dma_linux_par *dma;

	if (d == NULL || *d == NULL)
		return;
	dma = (struct aml_dma_linux_par *)(*d)->data;
	for (size_t i = 0; i < aml_vector_size(dma->data.requests); i++) {
		struct aml_dma_linux_par_request_data *req;

		req = aml_vector_get(dma->data.requests, i);
		if (req->type != AML_DMA_REQUEST_TYPE_INVALID) {
			pthread_cancel(req->thread);
			pthread_join(req->thread, NULL);
		}
		if (req->type == AML_DMA_REQUEST_TYPE_PTR) {
			aml_layout_dense_destroy(&req->dest);
			aml_layout_dense_destroy(&req->src);
		}
	}
	aml_vector_destroy(&dma->data.requests);
	pthread_mutex_destroy(&dma->data.lock);
	free(*d);
	*d = NULL;
}
