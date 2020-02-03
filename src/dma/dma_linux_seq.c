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

int aml_dma_request_linux_seq_copy_init(struct aml_dma_request_linux_seq *req,
					struct aml_layout *dest,
					struct aml_layout *src,
					aml_dma_operator op,
					void *op_arg)
{
	assert(req != NULL);
	req->type = AML_DMA_REQUEST_TYPE_LAYOUT;
	req->dest = dest;
	req->src = src;
	req->op = op;
	req->op_arg = op_arg;
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
	assert(req->op != NULL);
	return req->op(req->dest, req->src, req->op_arg);
}

struct aml_dma_linux_seq_inner_ops aml_dma_linux_seq_inner_ops = {
	aml_dma_linux_seq_do_copy,
};

/*******************************************************************************
 * Public API
 ******************************************************************************/

int aml_dma_linux_seq_create_request(struct aml_dma_data *d,
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
	struct aml_dma_linux_seq *dma =
		(struct aml_dma_linux_seq *)d;
	struct aml_dma_request_linux_seq *req;

	if (op == NULL)
		op = dma->data.default_op;
	if (op_arg == NULL)
		op_arg = dma->data.default_op_arg;

	pthread_mutex_lock(&dma->data.lock);
	req = aml_vector_add(dma->data.requests);
	aml_dma_request_linux_seq_copy_init(req, dest, src, op, op_arg);
	pthread_mutex_unlock(&dma->data.lock);
	*r = (struct aml_dma_request *)req;
	return 0;
}

int aml_dma_linux_seq_destroy_request(struct aml_dma_data *d,
				      struct aml_dma_request **r)
{
	assert(d != NULL); assert(r != NULL);
	struct aml_dma_linux_seq *dma =
		(struct aml_dma_linux_seq *)d;
	struct aml_dma_request_linux_seq *req;

	if (*r == NULL)
		return -AML_EINVAL;
	req = (struct aml_dma_request_linux_seq *)*r;

	aml_dma_request_linux_seq_copy_destroy(req);
	pthread_mutex_lock(&dma->data.lock);
	aml_vector_remove(dma->data.requests, req);
	pthread_mutex_unlock(&dma->data.lock);
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

	if (*r == NULL)
		return -AML_EINVAL;
	req = (struct aml_dma_request_linux_seq *)*r;

	/* execute */
	if (req->type != AML_DMA_REQUEST_TYPE_INVALID)
		dma->ops.do_copy(&dma->data, req);

	/* destroy a completed request */
	aml_dma_linux_seq_destroy_request(d, r);
	return 0;
}

int aml_dma_linux_seq_fprintf(const struct aml_dma_data *data,
			      FILE *stream, const char *prefix)
{
	const struct aml_dma_linux_seq *d;
	size_t vsize;

	fprintf(stream, "%s: dma-linux-seq: %p:\n", prefix, (void *)data);
	if (data == NULL)
		return AML_SUCCESS;

	d = (const struct aml_dma_linux_seq *)data;

	vsize = aml_vector_size(d->data.requests);
	/* ugly cast because ISO C forbids function pointer to void * */
	fprintf(stream, "%s: op: %p\n", prefix,
		(void *) (intptr_t) d->data.default_op);
	fprintf(stream, "%s: op-arg: %p\n", prefix, d->data.default_op_arg);
	fprintf(stream, "%s: requests: %zu\n", prefix, vsize);
	for (size_t i = 0; i < vsize; i++) {
		const struct aml_dma_request_linux_seq *r;

		r = aml_vector_get(d->data.requests, i);
		fprintf(stream, "%s: type: %d\n", prefix, r->type);
		if (r->type == AML_DMA_REQUEST_TYPE_INVALID)
			continue;

		fprintf(stream, "%s: layout-dest: %p\n", prefix,
			(void *)r->dest);
		aml_layout_fprintf(stream, prefix, r->dest);
		fprintf(stream, "%s: layout-src: %p\n", prefix, (void *)r->src);
		aml_layout_fprintf(stream, prefix, r->src);
		fprintf(stream, "%s: op: %p\n", prefix,
			(void *) (intptr_t)r->op);
		fprintf(stream, "%s: op-arg: %p\n", prefix, (void *)r->op_arg);
	}
	return AML_SUCCESS;
}

struct aml_dma_ops aml_dma_linux_seq_ops = {
	aml_dma_linux_seq_create_request,
	aml_dma_linux_seq_destroy_request,
	aml_dma_linux_seq_wait_request,
	aml_dma_linux_seq_fprintf,
};

/*******************************************************************************
 * Init functions:
 ******************************************************************************/

int aml_dma_linux_seq_create(struct aml_dma **dma, size_t nbreqs,
			     aml_dma_operator op, void *op_arg)
{
	struct aml_dma *ret = NULL;
	struct aml_dma_linux_seq *d;

	if (dma == NULL)
		return -AML_EINVAL;

	*dma = NULL;

	ret = AML_INNER_MALLOC(struct aml_dma, struct aml_dma_linux_seq);
	if (ret == NULL)
		return -AML_ENOMEM;

	ret->data = AML_INNER_MALLOC_GET_FIELD(ret, 2,
					       struct aml_dma,
					       struct aml_dma_linux_seq);
	ret->ops = &aml_dma_linux_seq_ops;
	d = (struct aml_dma_linux_seq *)ret->data;

	d->ops = aml_dma_linux_seq_inner_ops;

	if (op == NULL) {
		op = aml_copy_layout_generic;
		op_arg = NULL;
	}
	d->data.default_op = op;
	d->data.default_op_arg = op_arg;

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
	if (d == NULL)
		return;

	assert(d->data != NULL);
	l = (struct aml_dma_linux_seq *)d->data;
	aml_vector_destroy(&l->data.requests);
	pthread_mutex_destroy(&l->data.lock);
	free(d);
	*dma = NULL;
}
