/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include "aml.h"

#include "aml/dma/linux.h"

void aml_dma_linux_exec_request(struct aml_task_in *input,
                                struct aml_task_out *output)
{
	int err;
	struct aml_dma_linux_task_in *in =
	        (struct aml_dma_linux_task_in *)input;
	err = in->op(in->dst, in->src, in->op_arg);
	*(int *)output = err;
}

int aml_dma_linux_request_create(struct aml_dma_data *data,
                                 struct aml_dma_request **req,
                                 struct aml_layout *dest,
                                 struct aml_layout *src,
                                 aml_dma_operator op,
                                 void *op_arg)
{
	struct aml_sched *sched = (struct aml_sched *)data;
	struct aml_dma_linux_request *r = malloc(sizeof(*r));

	if (r == NULL)
		return -AML_ENOMEM;

	r->task.in = (struct aml_task_in *)&r->task_in;
	r->task.out = (struct aml_task_out *)&r->task_out;
	r->task.fn = aml_dma_linux_exec_request;
	r->task.data = req;

	r->task_in.dst = dest;
	r->task_in.src = src;
	r->task_in.op = op;
	r->task_in.op_arg = op_arg;
	r->task_in.req = r;
	r->task_out = AML_SUCCESS;

	if (req != NULL)
		*req = (struct aml_dma_request *)r;
	else
		r->flags = AML_DMA_LINUX_REQUEST_FLAGS_OWNED;
	return aml_sched_submit_task(sched, &r->task);
}

int aml_dma_linux_request_wait(struct aml_dma_data *dma,
                               struct aml_dma_request **req)
{
	struct aml_sched *sched = (struct aml_sched *)dma;
	struct aml_dma_linux_request *r = *(struct aml_dma_linux_request **)req;
	int out;

	int err = aml_sched_wait_task(sched, &r->task);
	if (err != AML_SUCCESS)
		return err;
	out = r->task_out;
	free(r);
	*req = NULL;
	return out;
}

int aml_dma_linux_request_wait_all(struct aml_dma_data *dma)
{
	struct aml_sched *sched = (struct aml_sched *)dma;
	struct aml_task *t = aml_sched_wait_any(sched);
	struct aml_dma_linux_task_in *input;
	int out;

	while (t != NULL) {
		input = (struct aml_dma_linux_task_in *)t->in;
		out = input->req->task_out;
		if (input->req->flags & AML_DMA_LINUX_REQUEST_FLAGS_OWNED)
			free(input->req);
		if (out != AML_SUCCESS)
			return out;
		t = aml_sched_wait_any(sched);
	}
	return AML_SUCCESS;
}

int aml_dma_linux_request_destroy(struct aml_dma_data *dma,
                                  struct aml_dma_request **req)
{
	(void)dma;
	free(*req);
	*req = NULL;
	return AML_SUCCESS;
}

int aml_dma_linux_create(struct aml_dma **dma, const size_t num_threads)
{
	struct aml_dma *d = malloc(sizeof(*d));
	if (d == NULL)
		return -AML_ENOMEM;
	d->data = (struct aml_dma_data *)aml_queue_sched_create(num_threads);
	if (d->data == NULL) {
		free(d);
		return -AML_ENOMEM;
	}
	d->ops = &aml_dma_linux_ops;

	*dma = d;
	return AML_SUCCESS;
}

int aml_dma_linux_destroy(struct aml_dma **dma)
{
	struct aml_sched *sched = (struct aml_sched *)(*dma)->data;
	struct aml_task *t = aml_sched_wait_any(sched);
	struct aml_dma_linux_task_in *input;

	while (t != NULL) {
		input = (struct aml_dma_linux_task_in *)t->in;
		free(input->req);
		t = aml_sched_wait_any(sched);
	}

	aml_queue_sched_destroy(&sched);
	free(*dma);
	*dma = NULL;
	return AML_SUCCESS;
}

int aml_dma_linux_copy_1D(struct aml_layout *dst,
                          const struct aml_layout *src,
                          void *arg)
{
	int err;

	(void)arg;
	const void *src_ptr = aml_layout_rawptr(src);
	void *dst_ptr = aml_layout_rawptr(dst);
	size_t n = 0;
	size_t size = 0;

	err = aml_layout_dims(src, &n);
	if (err != AML_SUCCESS)
		return err;
	size = aml_layout_element_size(src) * n;

	memcpy(dst_ptr, src_ptr, size);
	return AML_SUCCESS;
}

struct aml_dma_ops aml_dma_linux_ops = {
        .create_request = aml_dma_linux_request_create,
        .destroy_request = aml_dma_linux_request_destroy,
        .wait_request = aml_dma_linux_request_wait,
        .wait_all = aml_dma_linux_request_wait_all,
        .fprintf = NULL,
};
