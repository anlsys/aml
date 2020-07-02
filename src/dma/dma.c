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
#include "aml/layout/native.h"

#include <assert.h>

/*******************************************************************************
 * Generic DMA API:
 * Most of the stuff is dispatched to a different layer, using type-specific
 * functions.
 *
 * Note that the API is slightly different than the functions bellow, as we
 * abstract the request creation after this layer.
 ******************************************************************************/

int aml_dma_copy(struct aml_dma *dma, struct aml_layout *dest,
		 struct aml_layout *src, aml_dma_operator op, void *op_arg)
{
	int ret;
	struct aml_dma_request *req;

	if (dma == NULL || dest == NULL || src == NULL)
		return -AML_EINVAL;

	ret = dma->ops->create_request(dma->data, &req, dest, src, op, op_arg);
	if (ret != AML_SUCCESS)
		return ret;
	ret = dma->ops->wait_request(dma->data, &req);
	return ret;
}

int aml_dma_async_copy(struct aml_dma *dma, struct aml_dma_request **req,
		       struct aml_layout *dest, struct aml_layout *src,
		       aml_dma_operator op, void *op_arg)
{
	if (dma == NULL || req == NULL || dest == NULL || src == NULL)
		return -AML_EINVAL;

	return dma->ops->create_request(dma->data, req, dest, src, op, op_arg);
}

int aml_dma_cancel(struct aml_dma *dma, struct aml_dma_request **req)
{
	if (dma == NULL || req == NULL)
		return -AML_EINVAL;
	return dma->ops->destroy_request(dma->data, req);
}

int aml_dma_wait(struct aml_dma *dma, struct aml_dma_request **req)
{
	if (dma == NULL || req == NULL)
		return -AML_EINVAL;
	return dma->ops->wait_request(dma->data, req);
}

int aml_dma_fprintf(FILE *stream, const char *prefix,
		    const struct aml_dma *dma)
{
	assert(dma != NULL && dma->ops != NULL && stream != NULL);

	const char *p = (prefix == NULL) ? "" : prefix;

	return dma->ops->fprintf(dma->data, stream, p);
}
