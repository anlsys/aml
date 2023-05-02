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

#include "aml/dma/multiplex.h"

int aml_dma_multiplex_request_create(struct aml_dma_data *data,
                                     struct aml_dma_request **req,
                                     struct aml_layout *dest,
                                     struct aml_layout *src,
                                     aml_dma_operator op,
                                     void *op_arg)
{
	int err;
	struct aml_dma_multiplex_request *m_req = NULL;
	struct aml_dma_multiplex_data *m_data =
	        (struct aml_dma_multiplex_data *)data;

	if (op == NULL)
		op = aml_dma_multiplex_copy_single;

	/* the request is a bit different than usual for a dma, since its the
	 * operators that actually enqueue new requests.
	 *
	 * We end up not allocating the request here, just letting the operator
	 * know about the argument.
	 *
	 * Note that we still need to indicate whether the user wants a request
	 * or not.
	 */
	struct aml_dma_multiplex_copy_args mc_args = {
	        .m_data = m_data,
	        .m_req = req ? &m_req : NULL,
	        .args = op_arg,
	};
	err = op(dest, src, &mc_args);
	if (req)
		*req = (struct aml_dma_request *)m_req;
	return err;
}

int aml_dma_multiplex_request_wait(struct aml_dma_data *dma,
                                   struct aml_dma_request **req)
{
	struct aml_dma_multiplex_request *m_req =
	        (struct aml_dma_multiplex_request *)(*req);
	(void)dma;

	/* just loop over the dmas and wait for all of them. Technically there
	 * are cases where we should care about the order in which we do these
	 * waits (if some depend on the CPU to do the work), but we choose not
	 * to bother.
	 *
	 * Can be fixed by the user with careful ordering at dma_create time.
	 */
	int err = AML_SUCCESS;
	for (size_t i = 0; i < m_req->count; i++)
		err = err || aml_dma_wait(m_req->dmas[i], &(m_req->reqs[i]));

	free(m_req);
	*req = NULL;
	return err;
}

int aml_dma_multiplex_barrier(struct aml_dma_data *dma)
{
	struct aml_dma_multiplex_data *m_data =
	        (struct aml_dma_multiplex_data *)dma;

	/* just loop over the dmas and barrier for all of them.
	 *
	 * There's a problem with this, as we might wait on requests that where
	 * not initiated by us (someone sent a request directly in one of the
	 * inner dmas).
	 *
	 * We'll ignore that problem for now.
	 */
	int err = AML_SUCCESS;
	for (size_t i = 0; i < m_data->count; i++)
		err = err || aml_dma_barrier(m_data->dmas[i]);

	return err;
}

int aml_dma_multiplex_request_destroy(struct aml_dma_data *dma,
                                      struct aml_dma_request **req)
{
	(void)dma;
	if (req != NULL && *req != NULL) {
		struct aml_dma_multiplex_request *m_req =
		        (struct aml_dma_multiplex_request *)(*req);
		for (size_t i = 0; i < m_req->count; i++)
			aml_dma_cancel(m_req->dmas[i], &m_req->reqs[i]);
		free(m_req);
		*req = NULL;
	}
	return AML_SUCCESS;
}

int aml_dma_multiplex_create(struct aml_dma **dma,
                             const size_t num,
                             const struct aml_dma **dmas,
                             const size_t *weights)
{
	struct aml_dma *out = NULL;
	struct aml_dma_multiplex_data *data;

	if (dma == NULL || num == 0)
		return -AML_EINVAL;
	out = AML_INNER_MALLOC_EXTRA(num, struct aml_dma *,
	                             num * sizeof(size_t), struct aml_dma,
	                             struct aml_dma_multiplex_data);
	if (out == NULL)
		return -AML_ENOMEM;

	data = AML_INNER_MALLOC_GET_FIELD(out, 2, struct aml_dma,
	                                  struct aml_dma_multiplex_data);
	out->data = (struct aml_dma_data *)data;
	out->ops = &aml_dma_multiplex_ops;

	data->count = num;
	data->dmas = AML_INNER_MALLOC_GET_ARRAY(out, struct aml_dma *,
	                                        struct aml_dma,
	                                        struct aml_dma_multiplex_data);
	data->weights = AML_INNER_MALLOC_GET_EXTRA(
	        out, num, struct aml_dma *, struct aml_dma,
	        struct aml_dma_multiplex_data);

	memcpy(data->dmas, dmas, num * sizeof(struct aml_dma *));
	memcpy(data->weights, weights, num * sizeof(size_t));

	*dma = out;
	return AML_SUCCESS;
}

int aml_dma_multiplex_destroy(struct aml_dma **dma)
{
	if (dma == NULL)
		return AML_SUCCESS;
	free(*dma);
	*dma = NULL;
	return AML_SUCCESS;
}

int aml_dma_multiplex_copy_single(struct aml_layout *dst,
                                  const struct aml_layout *src,
                                  void *arg)
{
	int err;
	struct aml_dma_multiplex_copy_args *args =
	        (struct aml_dma_multiplex_copy_args *)arg;

	struct aml_dma_multiplex_data *m_data = args->m_data;
	size_t dma_idx = m_data->index;
	struct aml_dma *dma = m_data->dmas[dma_idx];
	aml_dma_operator op = args->args->ops[dma_idx];
	void *op_arg = args->args->op_args[dma_idx];

	struct aml_dma_request **inner_req = NULL;
	if (args->m_req != NULL) {
		struct aml_dma_multiplex_request *req = AML_INNER_MALLOC_ARRAY(
		        2, void *, struct aml_dma_multiplex_request);
		req->count = 1;
		req->reqs = AML_INNER_MALLOC_GET_ARRAY(
		        req, void *, struct aml_dma_multiplex_request);
		req->dmas = (struct aml_dma **)req->reqs + 1;
		req->dmas[0] = dma;
		*args->m_req = req;
		inner_req = &(req->reqs[0]);
	}
	err = aml_dma_async_copy_custom(dma, inner_req, dst,
	                                (struct aml_layout *)src, op, op_arg);
	if (err != AML_SUCCESS && args->m_req != NULL)
		free(*args->m_req);

	m_data->round++;
	if (m_data->round > m_data->weights[dma_idx]) {
		m_data->index = (dma_idx + 1) % m_data->count;
		m_data->round = 0;
	}
	return err;
}

struct aml_dma_ops aml_dma_multiplex_ops = {
        .create_request = aml_dma_multiplex_request_create,
        .destroy_request = aml_dma_multiplex_request_destroy,
        .wait_request = aml_dma_multiplex_request_wait,
        .barrier = aml_dma_multiplex_barrier,
        .fprintf = NULL,
};
