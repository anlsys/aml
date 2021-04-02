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
#include "aml/layout/dense.h"

// Data
#define size (1 << 24) // 16 MB
void *host_data;
void *device_data;

// Layout
const size_t ndims = 1;
const size_t dims = size;
const size_t element_size = 1;
const size_t stride = 0;
const size_t pitch = 1;
struct aml_layout *host_layout;
struct aml_layout *device_layout;

// Dma
struct aml_dma *dma;

// Ze
ze_context_handle_t context;
ze_driver_handle_t driver;
ze_device_handle_t device;
ze_command_list_handle_t cmd_list;

void setup()
{
	uint32_t ze_count = 1;

	// Data
	host_data = malloc(size);
	assert(host_data != NULL);

	assert(zeDriverGet(&ze_count, &driver) == ZE_RESULT_SUCCESS);
	assert(ze_count == 1);
	assert(zeDeviceGet(driver, &ze_count, &device) == ZE_RESULT_SUCCESS);
	assert(ze_count == 1);

	// Create context
	ze_context_desc_t context_desc = {
	        .stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC,
	        .pNext = NULL,
	        .flags = 0};
	assert(zeContextCreate(driver, &context_desc, &context) ==
	       ZE_RESULT_SUCCESS);

	// Alloc Device
	ze_device_mem_alloc_desc_t device_desc = {
	        .stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
	        .pNext = NULL,
	        .flags = ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_UNCACHED,
	        .ordinal = 0,
	};
	assert(zeMemAllocDevice(context, &device_desc, size, 64, device,
	                        &device_data) == ZE_RESULT_SUCCESS);

	// Init command list
	ze_command_queue_desc_t queue_desc = {
	        .stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
	        .pNext = NULL,
	        .ordinal = 0,
	        .index = 0,
	        .flags = ZE_COMMAND_QUEUE_FLAG_EXPLICIT_ONLY,
	        .mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,
	        .priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
	assert(zeCommandListCreateImmediate(context, device, &queue_desc,
	                                    &cmd_list) == ZE_RESULT_SUCCESS);

	// Layout
	assert(aml_layout_dense_create(&host_layout, host_data,
	                               AML_LAYOUT_ORDER_COLUMN_MAJOR,
	                               element_size, ndims, &dims, &stride,
	                               &pitch) == AML_SUCCESS);
	assert(aml_layout_dense_create(&device_layout, device_data,
	                               AML_LAYOUT_ORDER_COLUMN_MAJOR,
	                               element_size, ndims, &dims, &stride,
	                               &pitch) == AML_SUCCESS);

	// Dma
	assert(aml_dma_ze_create(&dma, device, 2) == AML_SUCCESS);
}

void teardown()
{
	// Data
	free(host_data);
	zeMemFree(context, device_data);

	// Command list
	zeCommandListDestroy(cmd_list);

	// Layout
	aml_layout_destroy(&host_layout);
	aml_layout_destroy(&device_layout);

	// Dma
	aml_dma_ze_destroy(&dma);
}

// Fill host_data and device_data
void initialize_buffers(int host_val, int device_val)
{
	memset(host_data, host_val, size);
	assert(zeCommandListAppendMemoryFill(cmd_list, device_data,
	                                     (void *)(&device_val),
	                                     sizeof(device_val), size, NULL, 0,
	                                     NULL) == ZE_RESULT_SUCCESS);
}

void test_sequential()
{
	void *check = malloc(size);

	// Sequential Dma from device to host.
	initialize_buffers(1, 0);
	assert(aml_dma_copy_custom(dma, host_layout, device_layout, NULL,
	                           NULL) == AML_SUCCESS);
	memset(check, 0, size);
	assert(!memcmp(check, host_data, size));
	free(check);
}

void test_async()
{
	struct aml_dma_request *req0, *req1;
	void *check = malloc(size);
	memset(check, 0, size);
	initialize_buffers(1, 0);
	// Queue to async copy to make sure dma engine handles multiple copies
	// and run them in order.
	assert(aml_dma_async_copy_custom(dma, &req0, host_layout, device_layout,
	                                 NULL, NULL) == AML_SUCCESS);
	assert(aml_dma_async_copy_custom(dma, &req1, device_layout, host_layout,
	                                 NULL, NULL) == AML_SUCCESS);
	assert(aml_dma_wait(dma, &req0) == AML_SUCCESS);
	assert(aml_dma_wait(dma, &req1) == AML_SUCCESS);
	assert(!memcmp(check, host_data, size));
	free(check);
}

int main(int argc, char **argv)
{
	assert(aml_init(&argc, &argv) == AML_SUCCESS);
	if (!aml_support_backends(AML_BACKEND_ZE))
		return 77;
	setup();

	test_sequential();
	test_async();

	teardown();
	aml_finalize();
}
