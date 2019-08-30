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

int aml_dma_request_linux_par_copy_init(struct aml_dma_request_linux_par *req,
					struct aml_layout *dest,
					struct aml_layout *src,
					aml_dma_operator op, void *op_arg)
{
	assert(req != NULL);
	req->type = AML_DMA_REQUEST_TYPE_LAYOUT;
	req->dest = dest;
	req->src = src;
	req->op = op;
	req->op_arg = op_arg;
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
	struct aml_dma_request_linux_par *req =
		(struct aml_dma_request_linux_par *)arg;

	pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, NULL);
	if (req->type != AML_DMA_REQUEST_TYPE_INVALID)
		req->op(req->dest, req->src, req->op_arg);
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
				     struct aml_layout *dest,
				     struct aml_layout *src,
				     aml_dma_operator op, void *op_arg)
{
	/* NULL checks done by the generic API */
	assert(d != NULL);
	assert(r != NULL);
	assert(dest != NULL);
	assert(src != NULL);
	struct aml_dma_linux_par *dma =
		(struct aml_dma_linux_par *)d;
	struct aml_dma_request_linux_par *req;

	if (op == NULL)
		op = dma->data.default_op;
	if (op_arg == NULL)
		op_arg = dma->data.default_op_arg;

	pthread_mutex_lock(&dma->data.lock);
	req = aml_vector_add(dma->data.requests);
	aml_dma_request_linux_par_copy_init(req, dest, src, op, op_arg);
	pthread_mutex_unlock(&dma->data.lock);
	pthread_create(&req->thread, NULL, dma->ops.do_thread, req);
	*r = (struct aml_dma_request *)req;
	return 0;
}

int aml_dma_linux_par_destroy_request(struct aml_dma_data *d,
				      struct aml_dma_request **r)
{
	assert(d != NULL);
	assert(r != NULL);
	struct aml_dma_linux_par *dma =
		(struct aml_dma_linux_par *)d;
	struct aml_dma_request_linux_par *req;

	if (*r == NULL)
		return -AML_EINVAL;
	req = (struct aml_dma_request_linux_par *)*r;

	/* we cancel and join, instead of killing, for a cleaner result */
	if (req->type != AML_DMA_REQUEST_TYPE_INVALID) {
		pthread_cancel(req->thread);
		pthread_join(req->thread, NULL);
	}

	/* make sure to destroy layouts before putting the request back in the
	 * vector
	 */
	aml_dma_request_linux_par_copy_destroy(req);
	pthread_mutex_lock(&dma->data.lock);
	aml_vector_remove(dma->data.requests, req);
	pthread_mutex_unlock(&dma->data.lock);
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

	if (*r == NULL)
		return -AML_EINVAL;
	req = (struct aml_dma_request_linux_par *)*r;

	if (req->type != AML_DMA_REQUEST_TYPE_INVALID)
		pthread_join(req->thread, NULL);

	aml_dma_request_linux_par_copy_destroy(req);
	pthread_mutex_lock(&dma->data.lock);
	aml_vector_remove(dma->data.requests, req);
	pthread_mutex_unlock(&dma->data.lock);
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

int aml_dma_linux_par_create(struct aml_dma **dma, size_t nbreqs,
			     aml_dma_operator op, void *op_arg)
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

	if (op == NULL) {
		op = aml_copy_layout_generic;
		op_arg = NULL;
	}
	d->data.default_op = op;
	d->data.default_op_arg = op_arg;

	/* allocate request array */
	aml_vector_create(&d->data.requests, nbreqs,
			  sizeof(struct aml_dma_request_linux_par),
			  offsetof(struct aml_dma_request_linux_par, type),
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
		struct aml_dma_request_linux_par *req;

		req = aml_vector_get(dma->data.requests, i);
		if (req->type != AML_DMA_REQUEST_TYPE_INVALID) {
			pthread_cancel(req->thread);
			pthread_join(req->thread, NULL);
		}
		aml_dma_request_linux_par_copy_destroy(req);
	}
	aml_vector_destroy(&dma->data.requests);
	pthread_mutex_destroy(&dma->data.lock);
	free(*d);
	*d = NULL;
}
