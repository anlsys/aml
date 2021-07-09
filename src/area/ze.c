/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *******************************************************************************/

#include "aml.h"

#include "aml/area/ze.h"
#include "aml/utils/backend/ze.h"

#define ZE(ze_call) aml_errno_from_ze_result(ze_call)

ze_host_mem_alloc_desc_t aml_ze_host_mem_alloc_desc = {
        .stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC,
        .pNext = NULL,
        .flags = 0,
};

int aml_area_ze_host_create(struct aml_area **area,
                            ze_host_mem_alloc_flag_t host_flags,
                            size_t alignment)
{
	int err;
	struct aml_area *out = NULL;
	struct aml_area_ze_data *data;

	if (aml_ze_default_data == NULL)
		return -AML_ENOTSUP;

	// Alloc area and set area fields.
	out = AML_INNER_MALLOC(struct aml_area, struct aml_area_ze_data);
	if (out == NULL)
		return -AML_ENOMEM;
	data = AML_INNER_MALLOC_GET_FIELD(out, 2, struct aml_area,
	                                  struct aml_area_ze_data);
	out->data = (struct aml_area_data *)data;

	err = ZE(zeContextCreate(aml_ze_default_data->driver,
	                         &aml_ze_context_desc, &data->context));
	if (err != AML_SUCCESS)
		goto err_with_area;

	// Initialize other fields.
	data->alignment = alignment;
	data->desc.host.host_desc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
	data->desc.host.host_desc.pNext = NULL;
	data->desc.host.host_desc.flags = host_flags;

	// Set op table:
	out->ops = &aml_area_ze_ops_host;

	*area = out;
	return AML_SUCCESS;
err_with_area:
	free(out);
	return err;
}

int aml_area_ze_device_create(struct aml_area **area,
                              ze_device_handle_t device,
                              uint32_t ordinal,
                              ze_device_mem_alloc_flag_t device_flags,
                              size_t alignment,
                              int flags)
{
	int err;
	struct aml_area *out = NULL;
	struct aml_area_ze_data *data;

	if (aml_ze_default_data == NULL)
		return -AML_ENOTSUP;

	// Alloc area and set area fields.
	out = AML_INNER_MALLOC(struct aml_area, struct aml_area_ze_data);
	if (out == NULL)
		return -AML_ENOMEM;
	data = AML_INNER_MALLOC_GET_FIELD(out, 2, struct aml_area,
	                                  struct aml_area_ze_data);
	out->data = (struct aml_area_data *)data;

	// Initialize device field
	data->desc.device.device = device;

	// Create context
	err = aml_ze_context_create(&data->context, device);
	if (err != AML_SUCCESS)
		goto err_with_area;

	// Initialize other fields.
	data->alignment = alignment;
	data->desc.device.device_desc.stype =
	        ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
	data->desc.device.device_desc.pNext = NULL;
	data->desc.device.device_desc.flags = device_flags;
	data->desc.device.device_desc.ordinal = ordinal;

	// Set op table:
	if (flags == AML_AREA_ZE_MMAP_SHARED_FLAGS)
		out->ops = &aml_area_ze_ops_shared;
	else
		out->ops = &aml_area_ze_ops_device;

	*area = out;
	return AML_SUCCESS;
err_with_area:
	free(out);
	return err;
}

void aml_area_ze_destroy(struct aml_area **area)
{
	if (area != NULL && *area != NULL) {
		struct aml_area_ze_data *data =
		        (struct aml_area_ze_data *)(*area)->data;
		aml_ze_context_destroy(data->context);
		free(*area);
		*area = NULL;
	}
}

int aml_area_ze_munmap(const struct aml_area_data *area_data,
                       void *ptr,
                       const size_t size)
{
	(void)size;
	struct aml_area_ze_data *data = (struct aml_area_ze_data *)area_data;
	return ZE(zeMemFree(data->context, ptr));
}

void *aml_area_ze_mmap_device(const struct aml_area_data *area_data,
                              size_t size,
                              struct aml_area_mmap_options *options)
{
	void *ptr;
	(void)options;
	struct aml_area_ze_data *data = (struct aml_area_ze_data *)area_data;
	int err = ZE(zeMemAllocDevice(
	        data->context, &data->desc.device.device_desc, size,
	        data->alignment, data->desc.device.device, &ptr));
	if (err != AML_SUCCESS) {
		aml_errno = err;
		return NULL;
	}
	return ptr;
}

struct aml_area_ops aml_area_ze_ops_device = {
        .mmap = aml_area_ze_mmap_device,
        .munmap = aml_area_ze_munmap,
        .fprintf = NULL,
};

void *aml_area_ze_mmap_shared(const struct aml_area_data *area_data,
                              size_t size,
                              struct aml_area_mmap_options *options)
{
	void *ptr;
	(void)options;
	ze_host_mem_alloc_desc_t host_desc = {
	        .stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC,
	        .pNext = NULL,
	        .flags = 0,
	};

	struct aml_area_ze_data *data = (struct aml_area_ze_data *)area_data;
	int err = ZE(zeMemAllocShared(
	        data->context, &data->desc.device.device_desc, &host_desc, size,
	        data->alignment, data->desc.device.device, &ptr));
	if (err != AML_SUCCESS) {
		aml_errno = err;
		return NULL;
	}
	return ptr;
}

struct aml_area_ops aml_area_ze_ops_shared = {
        .mmap = aml_area_ze_mmap_shared,
        .munmap = aml_area_ze_munmap,
        .fprintf = NULL,
};

void *aml_area_ze_mmap_host(const struct aml_area_data *area_data,
                            size_t size,
                            struct aml_area_mmap_options *options)
{
	void *ptr;
	(void)options;
	struct aml_area_ze_data *data = (struct aml_area_ze_data *)area_data;
	int err = ZE(zeMemAllocHost(data->context, &data->desc.host.host_desc,
	                            size, data->alignment, &ptr));
	if (err != AML_SUCCESS) {
		aml_errno = err;
		return NULL;
	}
	return ptr;
}

struct aml_area_ops aml_area_ze_ops_host = {
        .mmap = aml_area_ze_mmap_host,
        .munmap = aml_area_ze_munmap,
        .fprintf = NULL,
};
