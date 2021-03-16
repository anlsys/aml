/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *******************************************************************************/
#include "config.h"

#include <stdlib.h>
#include <string.h>

#include "aml.h"

#include "aml/area/ze.h"
#include "aml/utils/features.h"

const size_t sizes[4] = {1, 32, 4096, 1 << 20};

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
	               data->context, *data->device, &command_queue_desc,
	               &command_list) == ZE_RESULT_SUCCESS);
	return command_list;
}

void test_device_mmap()
{
	void *host_data;
	void *host_copy;
	void *device_data;
	struct aml_area *area;
	size_t ns = sizeof(sizes) / sizeof(*sizes);
	int err;
	size_t size;

	assert(aml_area_ze_create(&area, AML_AREA_ZE_MMAP_DEVICE_FLAGS, 0,
	                          ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_CACHED, NULL,
	                          64) == AML_SUCCESS);
	ze_command_list_handle_t command_list = get_command_list(area);

	host_data = malloc(sizes[ns - 1]);
	assert(host_data != NULL);
	host_copy = malloc(sizes[ns - 1]);
	assert(host_copy != NULL);

	for (size_t i = 0; i < ns; i++) {
		size = sizes[i];
		memset(host_data, 0, size);
		memset(host_copy, 1, size);

		device_data = aml_area_mmap(area, size, NULL);
		assert(device_data);

		err = zeCommandListAppendMemoryCopy(command_list, device_data,
		                                    host_data, size, NULL, 0,
		                                    NULL);
		assert(err == ZE_RESULT_SUCCESS);

		err = zeCommandListAppendMemoryCopy(command_list, host_copy,
		                                    device_data, size, NULL, 0,
		                                    NULL);
		assert(err == ZE_RESULT_SUCCESS);
		assert(!memcmp(host_data, host_copy, size));
		assert(aml_area_munmap(area, device_data, size) == AML_SUCCESS);
	}

	free(host_data);
	free(host_copy);

	zeCommandListDestroy(command_list);
	aml_area_ze_destroy(&area);
}

void test_host_mmap()
{
	void *host_data;
	void *host_copy;
	struct aml_area *area;
	size_t ns = sizeof(sizes) / sizeof(*sizes);
	size_t size;

	assert(aml_area_ze_create(&area, AML_AREA_ZE_MMAP_HOST_FLAGS,
	                          ZE_HOST_MEM_ALLOC_FLAG_BIAS_CACHED, 0, NULL,
	                          64) == AML_SUCCESS);
	host_copy = malloc(sizes[ns - 1]);
	assert(host_copy != NULL);

	for (size_t i = 0; i < ns; i++) {
		size = sizes[i];
		memset(host_copy, 1, size);
		host_data = aml_area_mmap(area, size, NULL);
		assert(host_data);
		memcpy(host_data, host_copy, size);
		assert(!memcmp(host_data, host_copy, size));
		assert(aml_area_munmap(area, host_data, size) == AML_SUCCESS);
	}

	free(host_copy);
	aml_area_ze_destroy(&area);
}

void test_shared_mmap()
{
	int err;
	void *unified_data;
	void *host_copy;
	struct aml_area *area;
	struct aml_area_ze_data *data;
	size_t ns = sizeof(sizes) / sizeof(*sizes);
	size_t size;

	// Data initialization
	host_copy = malloc(sizes[ns - 1]);
	assert(host_copy != NULL);

	// Map existing host data.
	assert(aml_area_ze_create(&area, AML_AREA_ZE_MMAP_SHARED_FLAGS,
	                          ZE_HOST_MEM_ALLOC_FLAG_BIAS_CACHED,
	                          ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_CACHED, NULL,
	                          64) == AML_SUCCESS);

	ze_command_list_handle_t command_list = get_command_list(area);

	data = (struct aml_area_ze_data *)area->data;
	for (size_t i = 0; i < ns; i++) {
		size = sizes[i];
		unified_data = aml_area_mmap(area, size, NULL);
		assert(unified_data != NULL);

		memset(unified_data, 0, size);
		memset(host_copy, 1, size);

		zeContextSystemBarrier(data->context, *data->device);

		err = zeCommandListAppendMemoryCopy(command_list, host_copy,
		                                    unified_data, size, NULL, 0,
		                                    NULL);
		assert(err == ZE_RESULT_SUCCESS);
		assert(!memcmp(unified_data, host_copy, size));

		memset(unified_data, 0, size);
		memset(host_copy, 1, size);
		zeContextSystemBarrier(data->context, *data->device);

		err = zeCommandListAppendMemoryCopy(command_list, unified_data,
		                                    host_copy, size, NULL, 0,
		                                    NULL);
		assert(err == ZE_RESULT_SUCCESS);
		assert(!memcmp(unified_data, host_copy, size));

		assert(!aml_area_munmap(area, unified_data, size));
	}

	zeCommandListDestroy(command_list);
	aml_area_ze_destroy(&area);
	free(host_copy);
}

int main(void)
{
	assert(aml_init(NULL, NULL) == AML_SUCCESS);
	if (!aml_support_backends(AML_BACKEND_ZE))
		return 77;
	test_device_mmap();
	test_host_mmap();
	test_shared_mmap();
	aml_finalize();
	return 0;
}
