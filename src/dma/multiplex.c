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
}

int aml_dma_multiplex_request_wait(struct aml_dma_data *dma,
                               struct aml_dma_request **req)
{
	struct aml_dma_multiplex_request *r = *(struct aml_dma_multiplex_request **)req;

	free(r);
	*req = NULL;
	return out;
}

int aml_dma_multiplex_barrier(struct aml_dma_data *dma)
{
	return AML_SUCCESS;
}

int aml_dma_multiplex_request_destroy(struct aml_dma_data *dma,
                                  struct aml_dma_request **req)
{
	(void)dma;
	free(*req);
	*req = NULL;
	return AML_SUCCESS;
}

int aml_dma_multiplex_create(struct aml_dma **dma, const size_t num,
		const struct aml_dma *dmas, const size_t *weigths)
{
	int err = AML_SUCCESS;
	struct aml_dma *out = NULL;
	struct aml_dma_multiplex_data *data;

	if (dma == NULL)
		return -AML_EINVAL;
	out = AML_INNER_MALLOC_EXTRA(num, struct aml_dma *,
			num * sizeof(size_t),
			struct aml_dma,
			struct aml_dma_multiplex_data);
	if (out == NULL)
		return -AML_ENOMEM;

	data = AML_INNER_MALLOC_GET_FIELD(out, 2, struct aml_dma,
			struct aml_dma_multiplex_data);
	out->data = data;
	out->ops = &aml_dma_multiplex_ops;

	data->dmas = AML_INNER_MALLOC_GET_ARRAY(out, struct aml_dma *,
			struct aml_dma, struct aml_dma_multiplex_data);
	data->weights = AML_INNER_MALLOC_GET_EXTRA(out, num, struct aml_dma *,
			struct aml_dma, struct aml_dma_multiplex_data);

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

int aml_dma_multiplex_copy_1D(struct aml_layout *dst,
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

	return AML_SUCCESS;
}

int aml_dma_multiplex_memcpy_op(struct aml_layout *dst,
                            const struct aml_layout *src,
                            void *arg)
{
	return AML_SUCCESS;
}

int aml_dma_multiplex_copy_chunks(struct aml_layout *dst,
                               const struct aml_layout *src,
                               void *arg)
{
	size_t d;
	size_t elem_size;
	(void)arg;

	return AML_SUCCESS;
}

struct aml_dma_ops aml_dma_multiplex_ops = {
        .create_request = aml_dma_multiplex_request_create,
        .destroy_request = aml_dma_multiplex_request_destroy,
        .wait_request = aml_dma_multiplex_request_wait,
        .barrier = aml_dma_multiplex_barrier,
        .fprintf = NULL,
};
