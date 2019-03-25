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
#include <assert.h>

/*******************************************************************************
 * Generic DMA API:
 * Most of the stuff is dispatched to a different layer, using type-specific
 * functions.
 *
 * Note that the API is slightly different than the functions bellow, as we
 * abstract the request creation after this layer.
 ******************************************************************************/

int aml_dma_copy(struct aml_dma *dma, ...)
{
	assert(dma != NULL);
	va_list ap;
	int ret;
	struct aml_dma_request *req;
	va_start(ap, dma);
	ret = dma->ops->create_request(dma->data, &req,
				       AML_DMA_REQUEST_TYPE_COPY, ap);
	va_end(ap);
	ret = dma->ops->wait_request(dma->data, req);
	return ret;
}

int aml_dma_async_copy(struct aml_dma *dma, struct aml_dma_request **req, ...)
{
	assert(dma != NULL);
	assert(req != NULL);
	va_list ap;
	int ret;
	va_start(ap, req);
	ret = dma->ops->create_request(dma->data, req,
				       AML_DMA_REQUEST_TYPE_COPY, ap);
	va_end(ap);
	return ret;
}

int aml_dma_cancel(struct aml_dma *dma, struct aml_dma_request *req)
{
	assert(dma != NULL);
	assert(req != NULL);
	return dma->ops->destroy_request(dma->data, req);
}

int aml_dma_wait(struct aml_dma *dma, struct aml_dma_request *req)
{
	assert(dma != NULL);
	assert(req != NULL);
	return dma->ops->wait_request(dma->data, req);
}
