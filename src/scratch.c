/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
*******************************************************************************/

#include <aml.h>
#include <assert.h>

/*******************************************************************************
 * Generic Scratchpad API:
 * Most of the stuff is dispatched to a different layer, using type-specific
 * functions.
 *
 * As for DMA, the API is slightly different than the functions below, as we
 * abstract the request creation after this layer.
 ******************************************************************************/

int aml_scratch_pull(struct aml_scratch *scratch, ...)
{
	assert(scratch != NULL);
	va_list ap;
	int ret;
	struct aml_scratch_request *req;
	va_start(ap, scratch);
	ret = scratch->ops->create_request(scratch->data, &req,
				       AML_SCRATCH_REQUEST_TYPE_PULL, ap);
	va_end(ap);
	ret = scratch->ops->wait_request(scratch->data, req);
	return ret;
}

int aml_scratch_async_pull(struct aml_scratch *scratch,
			   struct aml_scratch_request **req, ...)
{
	assert(scratch != NULL);
	assert(req != NULL);
	va_list ap;
	int ret;
	va_start(ap, req);
	ret = scratch->ops->create_request(scratch->data, req,
				       AML_SCRATCH_REQUEST_TYPE_PULL, ap);
	va_end(ap);
	return ret;
}

int aml_scratch_push(struct aml_scratch *scratch, ...)
{
	assert(scratch != NULL);
	struct aml_scratch_request *req;
	va_list ap;
	int ret;
	va_start(ap, scratch);
	ret = scratch->ops->create_request(scratch->data, &req,
				       AML_SCRATCH_REQUEST_TYPE_PUSH, ap);
	va_end(ap);
	ret = scratch->ops->wait_request(scratch->data, req);
	return ret;
}


int aml_scratch_async_push(struct aml_scratch *scratch, struct aml_scratch_request **req, ...)
{
	assert(scratch != NULL);
	assert(req != NULL);
	va_list ap;
	int ret;
	va_start(ap, req);
	ret = scratch->ops->create_request(scratch->data, req,
				       AML_SCRATCH_REQUEST_TYPE_PUSH, ap);
	va_end(ap);
	return ret;
}

int aml_scratch_cancel(struct aml_scratch *scratch, struct aml_scratch_request *req)
{
	assert(scratch != NULL);
	assert(req != NULL);
	return scratch->ops->destroy_request(scratch->data, req);
}

int aml_scratch_wait(struct aml_scratch *scratch, struct aml_scratch_request *req)
{
	assert(scratch != NULL);
	assert(req != NULL);
	return scratch->ops->wait_request(scratch->data, req);
}

void *aml_scratch_baseptr(const struct aml_scratch *scratch)
{
	assert(scratch != NULL);
	return scratch->ops->baseptr(scratch->data);
}

int aml_scratch_release(struct aml_scratch *scratch, int scratchid)
{
	assert(scratch != NULL);
	return scratch->ops->release(scratch->data, scratchid);
}
