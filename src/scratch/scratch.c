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
 * Generic Scratchpad API:
 * Most of the stuff is dispatched to a different layer, using type-specific
 * functions.
 *
 * As for DMA, the API is slightly different than the functions below, as we
 * abstract the request creation after this layer.
 ******************************************************************************/

int aml_scratch_pull(struct aml_scratch *scratch,
		     struct aml_layout **dest, int *scratchid,
		     struct aml_layout *src, int srcid)
{
	struct aml_scratch_request *req;
	int ret;

	if (scratch == NULL || dest == NULL || scratchid == NULL
	    || src == NULL)
		return -AML_EINVAL;
	
	ret = scratch->ops->create_request(scratch->data, &req,
					   AML_SCRATCH_REQUEST_TYPE_PULL,
					   dest, scratchid, src, srcid);
	if (ret)
		return ret;

	return scratch->ops->wait_request(scratch->data, req);
}

int aml_scratch_async_pull(struct aml_scratch *scratch,
			   struct aml_scratch_request **req,
			   struct aml_layout **dest, int *scratchid,
			   struct aml_layout *src, int srcid)
{
	if (scratch == NULL || dest == NULL || scratchid == NULL
	    || src == NULL)
		return -AML_EINVAL;

	return scratch->ops->create_request(scratch->data, req,
					    AML_SCRATCH_REQUEST_TYPE_PULL,
					    dest, scratchid, src, srcid);
}

int aml_scratch_push(struct aml_scratch *scratch,
		     struct aml_layout **dest, int *destid,
		     struct aml_layout *src, int srcid)
{
	struct aml_scratch_request *req;
	int ret;

	if (scratch == NULL || dest == NULL || destid == NULL
	    || src == NULL)
		return -AML_EINVAL;

	ret = scratch->ops->create_request(scratch->data, &req,
					   AML_SCRATCH_REQUEST_TYPE_PUSH,
					   dest, destid, src, srcid);
	if (ret)
		return ret;

	return scratch->ops->wait_request(scratch->data, req);
}


int aml_scratch_async_push(struct aml_scratch *scratch,
			   struct aml_scratch_request **req,
			   struct aml_layout **dest, int *destid,
			   struct aml_layout *src, int srcid)
{
	if (scratch == NULL || dest == NULL || destid == NULL
	    || src == NULL)
		return -AML_EINVAL;

	return scratch->ops->create_request(scratch->data, req,
					    AML_SCRATCH_REQUEST_TYPE_PUSH,
					    dest, destid, src, srcid);
}

int aml_scratch_cancel(struct aml_scratch *scratch,
		       struct aml_scratch_request *req)
{
	if (scratch == NULL || req == NULL)
		return -AML_EINVAL;
	return scratch->ops->destroy_request(scratch->data, req);
}

int aml_scratch_wait(struct aml_scratch *scratch,
		     struct aml_scratch_request *req)
{
	if (scratch == NULL || req == NULL)
		return -AML_EINVAL;
	return scratch->ops->wait_request(scratch->data, req);
}

int aml_scratch_release(struct aml_scratch *scratch, int scratchid)
{
	if (scratch == NULL)
		return -AML_EINVAL;
	return scratch->ops->release(scratch->data, scratchid);
}
