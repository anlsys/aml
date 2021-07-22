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
#include "aml/layout/native.h"

void aml_dma_linux_exec_request(struct aml_task_in *input,
                                struct aml_task_out *output)
{
	struct aml_dma_linux_task_in *in =
	        (struct aml_dma_linux_task_in *)input;
	*(int *)output = in->op(in->dst, in->src, in->op_arg);
	in->req->flags = in->req->flags | AML_DMA_LINUX_REQUEST_FLAGS_DONE;
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

	if (op == NULL)
		r->task_in.op = aml_dma_linux_copy_generic;
	else
		r->task_in.op = op;
	r->task_in.op_arg = op_arg;
	r->task_out = 0;

	r->task_in.req = r;

	if (req != NULL) {
		*req = (struct aml_dma_request *)r;
		r->flags = 0;
	} else
		r->flags = AML_DMA_LINUX_REQUEST_FLAGS_OWNED;

	return aml_sched_submit_task(sched, &r->task);
}

int aml_dma_linux_request_wait(struct aml_dma_data *dma,
                               struct aml_dma_request **req)
{
	struct aml_sched *sched = (struct aml_sched *)dma;
	struct aml_dma_linux_request *r = *(struct aml_dma_linux_request **)req;

	if (!(r->flags & AML_DMA_LINUX_REQUEST_FLAGS_DONE)) {
		int err = aml_sched_wait_task(sched, &r->task);
		if (err != AML_SUCCESS)
			return err;
	}
	int out = r->task_out;
	free(r);
	*req = NULL;
	return out;
}

int aml_dma_linux_barrier(struct aml_dma_data *dma)
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

int aml_dma_linux_memcpy_op(struct aml_layout *dst,
                            const struct aml_layout *src,
                            void *arg)
{
	memcpy(dst, src, (size_t)arg);
	return AML_SUCCESS;
}

static inline void aml_copy_layout_generic_helper(size_t d,
                                                  struct aml_layout *dst,
                                                  const struct aml_layout *src,
                                                  const size_t *elem_number,
                                                  size_t elem_size,
                                                  size_t *coords)
{
	if (d == 1) {
		for (size_t i = 0; i < elem_number[0]; i += 1) {
			coords[0] = i;
			memcpy(aml_layout_deref_native(dst, coords),
			       aml_layout_deref_native(src, coords), elem_size);
		}
	} else {
		for (size_t i = 0; i < elem_number[d - 1]; i += 1) {
			coords[d - 1] = i;
			aml_copy_layout_generic_helper(d - 1, dst, src,
			                               elem_number, elem_size,
			                               coords);
		}
	}
}

int aml_dma_linux_copy_generic(struct aml_layout *dst,
                               const struct aml_layout *src,
                               void *arg)
{
	size_t d;
	size_t elem_size;
	(void)arg;

	assert(aml_layout_ndims(dst) == aml_layout_ndims(src));
	d = aml_layout_ndims(dst);
	assert(aml_layout_element_size(dst) == aml_layout_element_size(src));
	elem_size = aml_layout_element_size(dst);

	size_t coords[d];
	size_t elem_number[d];
	size_t elem_number2[d];

	aml_layout_dims_native(src, elem_number);
	aml_layout_dims_native(dst, elem_number2);
	for (size_t i = 0; i < d; i += 1)
		assert(elem_number[i] == elem_number2[i]);
	aml_copy_layout_generic_helper(d, dst, src, elem_number, elem_size,
	                               coords);
	return 0;
}

struct aml_dma_ops aml_dma_linux_ops = {
        .create_request = aml_dma_linux_request_create,
        .destroy_request = aml_dma_linux_request_destroy,
        .wait_request = aml_dma_linux_request_wait,
        .barrier = aml_dma_linux_barrier,
        .fprintf = NULL,
};
