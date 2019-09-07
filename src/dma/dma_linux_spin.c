/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
*******************************************************************************/

#include "config.h"
#include "aml.h"
#include "aml/dma/linux-spin.h"
#include "aml/layout/dense.h"

#include <assert.h>
#include <errno.h>
#include <sys/mman.h>

#define ASMPAUSE asm("" : : : "memory")

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

int aml_dma_request_linux_spin_copy_init(struct aml_dma_request_linux_spin *req,
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

int aml_dma_request_linux_spin_copy_destroy(struct aml_dma_request_linux_spin *r)
{
	assert(r != NULL);
	r->type = AML_DMA_REQUEST_TYPE_INVALID;
	return 0;
}

/*******************************************************************************
 * Internal functions
 ******************************************************************************/

void *aml_dma_linux_spin_do_thread(void *arg)
{
	struct aml_dma_request_linux_spin *req =
		(struct aml_dma_request_linux_spin *)arg;

	pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, NULL);
	while(1) {
		pthread_spin_lock(&req->lock);
		if (req->type != AML_DMA_REQUEST_TYPE_INVALID) {
			req->op(req->dest, req->src, req->op_arg);
			req->type = AML_DMA_REQUEST_TYPE_INVALID;
		}
		pthread_spin_unlock(&req->lock);
	}
	return NULL;
}

struct aml_dma_linux_spin_ops aml_dma_linux_spin_inner_ops = {
	aml_dma_linux_spin_do_thread,
};

/*******************************************************************************
 * Public API
 ******************************************************************************/

int aml_dma_linux_spin_create_request(struct aml_dma_data *d,
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
	struct aml_dma_linux_spin *dma =
		(struct aml_dma_linux_spin *)d;
	struct aml_dma_request_linux_spin *req;
	req = &(dma->data.req);

	if (op == NULL)
		op = dma->data.default_op;
	if (op_arg == NULL)
		op_arg = dma->data.default_op_arg;

	pthread_spin_lock(&dma->data.req.lock);
	if (req->type != AML_DMA_REQUEST_TYPE_INVALID) {
		pthread_spin_unlock(&dma->data.req.lock);
		return -AML_EINVAL;
	}
	aml_dma_request_linux_spin_copy_init(req, dest, src, op, op_arg);
	pthread_spin_unlock(&dma->data.req.lock);
	*r = (struct aml_dma_request *)req;
	return 0;
}

int aml_dma_linux_spin_destroy_request(struct aml_dma_data *d,
				      struct aml_dma_request **r)
{
	return 0;
}

int aml_dma_linux_spin_wait_request(struct aml_dma_data *d,
				   struct aml_dma_request **r)
{
	assert(d != NULL);
	assert(r != NULL);
	struct aml_dma_linux_spin *dma = (struct aml_dma_linux_spin *)d;
	struct aml_dma_request_linux_spin *req;

	if (*r == NULL)
		return -AML_EINVAL;
	req = (struct aml_dma_request_linux_spin *)*r;

	while (1) {
		while (req->type != AML_DMA_REQUEST_TYPE_INVALID){ASMPAUSE;}
		pthread_spin_lock(&(req->lock));//
		if (req->type == AML_DMA_REQUEST_TYPE_INVALID) break;
		pthread_spin_unlock(&(req->lock));
	}
	pthread_spin_unlock(&(req->lock));

	*r = NULL;
	return 0;
}

struct aml_dma_ops aml_dma_linux_spin_ops = {
	aml_dma_linux_spin_create_request,
	aml_dma_linux_spin_destroy_request,
	aml_dma_linux_spin_wait_request,
};

/*******************************************************************************
 * Init functions:
 ******************************************************************************/

int aml_dma_linux_spin_create(struct aml_dma **dma, const cpu_set_t *cpuset,
			     aml_dma_operator op, void *op_arg)
{
	struct aml_dma *ret = NULL;
	struct aml_dma_linux_spin *d;

	if (dma == NULL)
		return -AML_EINVAL;

	*dma = NULL;

	ret = AML_INNER_MALLOC_2(struct aml_dma, struct aml_dma_linux_spin);
	if (ret == NULL)
		return -AML_ENOMEM;

	ret->data = AML_INNER_MALLOC_NEXTPTR(ret, struct aml_dma,
					     struct aml_dma_linux_spin);
	ret->ops = &aml_dma_linux_spin_ops;
	d = (struct aml_dma_linux_spin *)ret->data;
	d->ops = aml_dma_linux_spin_inner_ops;

	if (op == NULL) {
		op = aml_copy_layout_generic;
		op_arg = NULL;
	}
	d->data.default_op = op;
	d->data.default_op_arg = op_arg;

	/* allocate request array */
	d->data.req.type = AML_DMA_REQUEST_TYPE_INVALID;
	pthread_spin_init(&d->data.req.lock, PTHREAD_PROCESS_PRIVATE);

	pthread_create(&d->data.req.thread, NULL, d->ops.do_thread, &d->data.req);
	if (cpuset)
		pthread_setaffinity_np(d->data.req.thread, sizeof(cpu_set_t), cpuset);

	*dma = ret;
	return 0;
}

void aml_dma_linux_spin_destroy(struct aml_dma **d)
{
	struct aml_dma_linux_spin *dma;

	if (d == NULL || *d == NULL)
		return;
	dma = (struct aml_dma_linux_spin *)(*d)->data;
	struct aml_dma_request_linux_spin *req;

	req = &dma->data.req;
	if (req->type != AML_DMA_REQUEST_TYPE_INVALID) {
		pthread_cancel(req->thread);
		pthread_join(req->thread, NULL);
	}
	pthread_spin_destroy(&req->lock);
	free(*d);
	*d = NULL;
}
