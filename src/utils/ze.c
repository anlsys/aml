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

#include "aml/area/ze.h"
#include "aml/dma/ze.h"
#include "aml/utils/ze.h"

int aml_ze_data_create(struct aml_ze_data **out, uint32_t driver_num)
{
	uint32_t ze_count = 0;

	if (zeDriverGet(&ze_count, NULL) != ZE_RESULT_SUCCESS)
		return -AML_ENOTSUP;

	ze_driver_handle_t driver[ze_count];
	if (zeDriverGet(&ze_count, driver) != ZE_RESULT_SUCCESS)
		return -AML_ENOTSUP;
	if (driver_num >= ze_count)
		return -AML_EINVAL;

	uint32_t num_device = 0;
	if (zeDeviceGet(driver[driver_num], &num_device, NULL) !=
	    ZE_RESULT_SUCCESS)
		return -AML_ENOTSUP;

	ze_device_handle_t device[num_device];
	if (zeDeviceGet(driver[driver_num], &num_device, device) !=
	    ZE_RESULT_SUCCESS)
		return -AML_ENOTSUP;

	uint32_t num_subdevice[num_device];
	uint32_t tot_subdevice = 0;
	memset(num_subdevice, 0, sizeof(num_subdevice));

	for (uint32_t i = 0; i < num_device; i++) {
		if (zeDeviceGetSubDevices(device[i], num_subdevice + i, NULL) !=
		    ZE_RESULT_SUCCESS)
			return -AML_FAILURE;
		tot_subdevice += num_subdevice[i];
	}

	struct aml_ze_data *data;
	data = AML_INNER_MALLOC_EXTRA(tot_subdevice + num_device, uint32_t,
	                              (num_device + tot_subdevice) *
	                                      sizeof(ze_device_handle_t),
	                              struct aml_ze_data);

	if (data == NULL)
		return -AML_ENOMEM;

	ze_device_handle_t *device_handles = AML_INNER_MALLOC_GET_EXTRA(
	        data, tot_subdevice + num_device, uint32_t, struct aml_ze_data);
	uint32_t *nums =
	        AML_INNER_MALLOC_GET_ARRAY(data, uint32_t, struct aml_ze_data);

	data->driver = driver[driver_num];
	data->num_device = num_device;
	data->device = device_handles;
	memcpy(data->device, device, sizeof(device));
	data->num_subdevice = nums;
	memcpy(data->num_subdevice, num_subdevice, sizeof(num_subdevice));

	device_handles += num_device;
	nums += num_device;
	for (uint32_t i = 0; i < num_device; i++) {
		data->subdevice[i] = device_handles;
		if (zeDeviceGetSubDevices(device[i], num_subdevice + i,
		                          data->subdevice[i]) !=
		    ZE_RESULT_SUCCESS)
			goto error;
		device_handles += num_subdevice[i];

		data->num_memories[i] = nums;
		for (uint32_t j = 0; j < num_subdevice[i]; j++) {
			nums[j] = 0;
			if (zeDeviceGetMemoryProperties(data->subdevice[i][j],
			                                nums + j, NULL) !=
			    ZE_RESULT_SUCCESS)
				goto error;
		}
		nums += num_subdevice[i];
	}

	*out = data;
	return AML_SUCCESS;

error:
	free(data);
	return -AML_ENOTSUP;
}

int aml_ze_data_destroy(struct aml_ze_data **out)
{
	if (out == NULL || *out != NULL)
		return -AML_EINVAL;
	free(*out);
	*out = NULL;
	return AML_SUCCESS;
}

struct aml_ze_data *aml_ze_default_data;
struct aml_area *aml_area_ze_device;
struct aml_area *aml_area_ze_host;
struct aml_dma *aml_dma_ze_default;

int aml_ze_init(void)
{
	int err = AML_SUCCESS;

	if (!aml_support_backends(AML_BACKEND_ZE))
		goto error;

	err = aml_ze_data_create(&aml_ze_default_data, 0);
	if (err != AML_SUCCESS)
		goto error;

	err = aml_area_ze_device_create(&aml_area_ze_device,
	                                aml_ze_default_data->device[0], 0,
	                                ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_CACHED,
	                                64, AML_AREA_ZE_MMAP_DEVICE_FLAGS);
	if (err != ZE_RESULT_SUCCESS)
		goto error_with_data;

	err = aml_area_ze_host_create(&aml_area_ze_host,
	                              ZE_HOST_MEM_ALLOC_FLAG_BIAS_CACHED, 64);
	if (err != ZE_RESULT_SUCCESS)
		goto error_with_ze_area_device;

	err = aml_dma_ze_create(&aml_dma_ze_default,
	                        aml_ze_default_data->device[0], 256);
	if (err != ZE_RESULT_SUCCESS)
		goto error_with_ze_area_host;

	return AML_SUCCESS;

error_with_ze_area_host:
	aml_area_ze_destroy(&aml_area_ze_host);
error_with_ze_area_device:
	aml_area_ze_destroy(&aml_area_ze_device);

error_with_data:
	aml_ze_data_destroy(&aml_ze_default_data);
error:
	aml_ze_default_data = NULL;
	aml_area_ze_device = NULL;
	aml_area_ze_host = NULL;
	aml_dma_ze_default = NULL;

	return err;
}

int aml_ze_finalize(void)
{
	if (aml_area_ze_device != NULL)
		aml_area_ze_destroy(&aml_area_ze_device);
	if (aml_area_ze_host != NULL)
		aml_area_ze_destroy(&aml_area_ze_host);
	if (aml_dma_ze_default != NULL)
		aml_dma_ze_destroy(&aml_dma_ze_default);
	if (aml_ze_default_data != NULL)
		aml_ze_data_destroy(&aml_ze_default_data);
	return AML_SUCCESS;
}

ze_context_desc_t aml_ze_context_desc = {
        .stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC,
        .pNext = NULL,
        .flags = 0,
};
