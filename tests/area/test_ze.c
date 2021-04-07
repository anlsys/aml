/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *******************************************************************************/
#include <stdlib.h>
#include <string.h>

#include "aml.h"

#include "aml/area/ze.h"

const size_t sizes[4] = {1, 32, 4096, 1 << 20};
ze_driver_handle_t driver;
ze_device_handle_t device;

void setup()
{
	uint32_t ze_count = 1;
	assert(zeDriverGet(&ze_count, &driver) == ZE_RESULT_SUCCESS);
	ze_count = 1;
	assert(zeDeviceGet(driver, &ze_count, &device) == ZE_RESULT_SUCCESS);
}

ze_command_list_handle_t get_command_list(struct aml_area *area)
{
	ze_command_list_handle_t command_list;
	struct aml_area_ze_data *data;
	data = (struct aml_area_ze_data *)area->data;

	ze_command_queue_desc_t command_queue_desc = {
	        .stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES,
	        .pNext = NULL,
	        .ordinal = 0,
	        .index = 0,
	        .flags = ZE_COMMAND_LIST_FLAG_EXPLICIT_ONLY,
	        .mode = ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS,
	        .priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL,
	};

	assert(zeCommandListCreateImmediate(
	               data->context, data->desc.device.device,
	               &command_queue_desc,
	               &command_list) == ZE_RESULT_SUCCESS);
	return command_list;
}

void test_device_mmap(size_t size)
{
	void *host_data;
	void *host_copy;
	void *device_data;
	struct aml_area *area;

	assert(aml_area_ze_device_create(&area, device, 0, 0, 32,
	                                 AML_AREA_ZE_MMAP_DEVICE_FLAGS) ==
	       AML_SUCCESS);
	ze_command_list_handle_t command_list = get_command_list(area);

	// Alloc buffers
	host_data = malloc(size);
	assert(host_data != NULL);
	memset(host_data, 0, size);
	host_copy = malloc(size);
	assert(host_copy != NULL);
	memset(host_copy, 1, size);
	device_data = aml_area_mmap(area, size, NULL);
	assert(device_data != NULL);

	// Check copy to buffer allocated with aml area is working.
	assert(zeCommandListAppendMemoryCopy(command_list, device_data,
	                                     host_data, size, NULL, 0,
	                                     NULL) == ZE_RESULT_SUCCESS);

	assert(zeCommandListAppendMemoryCopy(command_list, host_copy,
	                                     device_data, size, NULL, 0,
	                                     NULL) == ZE_RESULT_SUCCESS);

	assert(!memcmp(host_data, host_copy, size));

	// Cleanup
	assert(aml_area_munmap(area, device_data, size) == AML_SUCCESS);
	free(host_data);
	free(host_copy);
	zeCommandListDestroy(command_list);
	aml_area_ze_destroy(&area);
}

void test_shared_mmap(size_t size)
{
	int err;
	void *unified_data;
	void *host_copy;
	struct aml_area *area;
	struct aml_area_ze_data *data;

	// Create area
	assert(aml_area_ze_device_create(&area, device, 0, 0, 32,
	                                 AML_AREA_ZE_MMAP_SHARED_FLAGS) ==
	       AML_SUCCESS);
	data = (struct aml_area_ze_data *)area->data;

	// Data initialization
	host_copy = malloc(size);
	assert(host_copy != NULL);
	memset(host_copy, 1, size);
	unified_data = aml_area_mmap(area, size, NULL);
	assert(unified_data != NULL);
	memset(unified_data, 0, size);

	// Ensure write are observable
	zeContextSystemBarrier(data->context, data->desc.device.device);

	ze_command_list_handle_t command_list = get_command_list(area);

	// Copy from device/shared buffer to host buffer
	assert(zeCommandListAppendMemoryCopy(command_list, host_copy,
	                                     unified_data, size, NULL, 0,
	                                     NULL) == ZE_RESULT_SUCCESS);
	assert(!memcmp(unified_data, host_copy, size));

	// Reinitialize data
	memset(unified_data, 0, size);
	memset(host_copy, 1, size);
	zeContextSystemBarrier(data->context, data->desc.device.device);

	// Copy to device/shared buffer from host buffer
	err = zeCommandListAppendMemoryCopy(command_list, unified_data,
	                                    host_copy, size, NULL, 0, NULL);
	assert(err == ZE_RESULT_SUCCESS);
	assert(!memcmp(unified_data, host_copy, size));

	// Cleanup
	assert(!aml_area_munmap(area, unified_data, size));
	zeCommandListDestroy(command_list);
	aml_area_ze_destroy(&area);
	free(host_copy);
}

int main(void)
{
	assert(aml_init(NULL, NULL) == AML_SUCCESS);
	if (!aml_support_backends(AML_BACKEND_ZE))
		return 77;
	setup();
	test_device_mmap(4096);
	test_shared_mmap(4096);
	aml_finalize();
	return 0;
}
