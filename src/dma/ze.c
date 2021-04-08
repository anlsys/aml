/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include "aml.h"

#include "aml/dma/ze.h"

int aml_errno_from_ze_result(ze_result_t err);
#define ZE(ze_call) aml_errno_from_ze_result(ze_call)

int aml_dma_ze_create(struct aml_dma **dma,
                      ze_device_handle_t device,
                      const uint32_t pool_size)
{
	int err = AML_SUCCESS;
	uint32_t count = 1;
	ze_driver_handle_t driver;
	struct aml_dma *out = NULL;
	struct aml_dma_ze_data *data;

	// Alloc dma
	out = AML_INNER_MALLOC(struct aml_dma, struct aml_dma_ze_data);
	if (out == NULL)
		return -AML_ENOMEM;
	data = AML_INNER_MALLOC_GET_FIELD(out, 2, struct aml_dma,
	                                  struct aml_dma_ze_data);
	out->data = (struct aml_dma_data *)data;
	out->ops = &aml_dma_ze_ops;

	// Get first driver
	err = ZE(zeDriverGet(&count, &driver));
	if (err != AML_SUCCESS)
		goto err_with_dma;

	// Set device field
	data->device = device;

	// Set context field
	ze_context_desc_t context_desc = {
	        .stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC,
	        .pNext = NULL,
	        .flags = 0};
	err = ZE(zeContextCreate(driver, &context_desc, &data->context));
	if (err != AML_SUCCESS)
		goto err_with_dma;

	// Prepare Command Queue
	ze_command_queue_desc_t queue_desc = {
	        .stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
	        .pNext = NULL,
	        .ordinal = 0,
	        .index = 0,
	        .flags = ZE_COMMAND_QUEUE_FLAG_EXPLICIT_ONLY,
	        .mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,
	        .priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
	err = ZE(zeCommandListCreateImmediate(
	        data->context, device, &queue_desc, &data->command_list));
	if (err != AML_SUCCESS)
		goto err_with_context;

	// Create event pool
	data->event_pool_size = pool_size;
	ze_event_pool_desc_t pool_desc = {
	        .stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC,
	        .pNext = NULL,
	        .flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE,
	        .count = pool_size,

	};
	err = ZE(zeEventPoolCreate(data->context, &pool_desc, 1, &device,
	                           &data->events));
	if (err != AML_SUCCESS)
		goto err_with_command_list;

	*dma = out;
	return AML_SUCCESS;
err_with_command_list:
	zeCommandListDestroy(data->command_list);
err_with_context:
	zeContextDestroy(data->context);
err_with_dma:
	free(out);
	return err;
}

int aml_dma_ze_destroy(struct aml_dma **dma)
{
	if (dma != NULL && *dma != NULL) {
		struct aml_dma_ze_data *data =
		        (struct aml_dma_ze_data *)(*dma)->data;
		zeEventPoolDestroy(data->events);
		zeCommandListDestroy(data->command_list);
		zeContextDestroy(data->context);
	}
	return AML_SUCCESS;
}

struct aml_dma_ze_copy_args {
	struct aml_dma_ze_data *ze_data;
	struct aml_dma_ze_request *ze_req;
};

int aml_dma_ze_copy_1D(struct aml_layout *dst,
                       const struct aml_layout *src,
                       void *arg)
{
	struct aml_dma_ze_copy_args *args = (struct aml_dma_ze_copy_args *)arg;
	size_t n = 0;
	size_t size = 0;
	int err;

	err = aml_layout_dims(src, &n);
	if (err != AML_SUCCESS)
		return err;
	size = aml_layout_element_size(src) * n;

	return ZE(zeCommandListAppendMemoryCopy(
	        args->ze_data->command_list, aml_layout_rawptr(dst),
	        aml_layout_rawptr(src), size, args->ze_req->event, 0, NULL));
}

int aml_dma_ze_request_create(struct aml_dma_data *data,
                              struct aml_dma_request **req,
                              struct aml_layout *dest,
                              struct aml_layout *src,
                              aml_dma_operator op,
                              void *op_arg)
{
	int err;
	struct aml_dma_ze_request *ze_req;
	struct aml_dma_ze_data *ze_data = (struct aml_dma_ze_data *)data;
	int request_flags = (int)(intptr_t)op_arg;

	if (op == NULL)
		op = aml_dma_ze_copy_1D;

	// Allocate space for the request.
	ze_req = AML_INNER_MALLOC(struct aml_dma_ze_request);
	if (ze_req == NULL)
		return -AML_ENOMEM;

	// Flags setting behaviour when request is waited for.
	ze_event_scope_flags_t wait_flag = 0;
	if (request_flags &
	    (AML_DMA_ZE_REQUEST_UNKNOWN | AML_DMA_ZE_REQUEST_HOST_DEVICE |
	     AML_DMA_ZE_REQUEST_DEVICE_DEVICE))
		wait_flag |= ZE_EVENT_SCOPE_FLAG_SUBDEVICE;
	if (request_flags &
	    (AML_DMA_ZE_REQUEST_UNKNOWN | AML_DMA_ZE_REQUEST_DEVICE_HOST))
		wait_flag |= ZE_EVENT_SCOPE_FLAG_HOST;
	else if (!request_flags) {
		wait_flag = ZE_EVENT_SCOPE_FLAG_HOST |
		            ZE_EVENT_SCOPE_FLAG_DEVICE;
	}

	// Create event.
	ze_event_desc_t desc = {
	        .stype = ZE_STRUCTURE_TYPE_EVENT_DESC,
	        .pNext = NULL,
	        .index = 0,
	        .signal = 0,
	        .wait = wait_flag,
	};
	err = ZE(zeEventCreate(ze_data->events, &desc, &ze_req->event));
	if (err != AML_SUCCESS)
		goto err_with_req;

	// Queue copy.
	struct aml_dma_ze_copy_args args = {
	        .ze_data = ze_data,
	        .ze_req = ze_req,
	};
	err = op(dest, src, &args);
	if (err != AML_SUCCESS)
		goto err_with_event;

	*req = (struct aml_dma_request *)ze_req;
	return AML_SUCCESS;
err_with_event:
	zeEventDestroy(ze_req->event);
err_with_req:
	free(ze_req);
	return err;
}

int aml_dma_ze_request_wait(struct aml_dma_data *data,
                            struct aml_dma_request **req)
{
	(void)data;
	struct aml_dma_ze_request *ze_req = (struct aml_dma_ze_request *)(*req);
	return ZE(zeEventHostSynchronize(ze_req->event, UINT64_MAX));
}

int aml_dma_ze_request_destroy(struct aml_dma_data *data,
                               struct aml_dma_request **req)
{
	(void)data;
	if (req != NULL && *req != NULL) {
		struct aml_dma_ze_request *ze_req =
		        (struct aml_dma_ze_request *)(*req);
		zeEventDestroy(ze_req->event);
		free(*req);
		*req = NULL;
	}
	return AML_SUCCESS;
}

struct aml_dma_ops aml_dma_ze_ops = {
        .create_request = aml_dma_ze_request_create,
        .destroy_request = aml_dma_ze_request_destroy,
        .wait_request = aml_dma_ze_request_wait,
        .fprintf = NULL,
};
