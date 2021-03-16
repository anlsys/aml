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

#include "aml/area/ze.h"
#include "aml/utils/error.h"

static inline int aml_errno_from_ze_result(ze_result_t err)
{
	switch (err) {
	case ZE_RESULT_SUCCESS:
		return AML_SUCCESS;
	case ZE_RESULT_ERROR_DEVICE_LOST:
	case ZE_RESULT_ERROR_MODULE_BUILD_FAILURE:
	case ZE_RESULT_ERROR_MODULE_LINK_FAILURE:
	case ZE_RESULT_ERROR_UNKNOWN:
	case ZE_RESULT_FORCE_UINT32:
		return AML_FAILURE;
	case ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY:
	case ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY:
		return AML_ENOMEM;
	case ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE:
	case ZE_RESULT_ERROR_UNINITIALIZED:
	case ZE_RESULT_ERROR_INVALID_ARGUMENT:
	case ZE_RESULT_ERROR_INVALID_NULL_HANDLE:
	case ZE_RESULT_ERROR_INVALID_NULL_POINTER:
	case ZE_RESULT_ERROR_INVALID_SIZE:
	case ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT:
	case ZE_RESULT_ERROR_INVALID_ENUMERATION:
	case ZE_RESULT_ERROR_INVALID_NATIVE_BINARY:
	case ZE_RESULT_ERROR_INVALID_GLOBAL_NAME:
	case ZE_RESULT_ERROR_INVALID_KERNEL_NAME:
	case ZE_RESULT_ERROR_INVALID_FUNCTION_NAME:
	case ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION:
	case ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION:
	case ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX:
	case ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE:
	case ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE:
	case ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED:
	case ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE:
		return AML_EINVAL;
	case ZE_RESULT_ERROR_OVERLAPPING_REGIONS:
		return AML_EDOM;
	case ZE_RESULT_ERROR_UNSUPPORTED_VERSION:
	case ZE_RESULT_ERROR_UNSUPPORTED_FEATURE:
	case ZE_RESULT_ERROR_UNSUPPORTED_SIZE:
	case ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT:
	case ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION:
	case ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT:
		return AML_ENOTSUP;
	case ZE_RESULT_NOT_READY:
	case ZE_RESULT_ERROR_NOT_AVAILABLE:
	case ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE:
		return AML_EBUSY;
	case ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS:
		return AML_EPERM;
	default:
		return AML_FAILURE;
	};
}
#define ZE(ze_call) aml_errno_from_ze_result(ze_call)

int aml_area_ze_create(struct aml_area **area,
                       unsigned long alloc_type,
                       ze_host_mem_alloc_flag_t host_flags,
                       ze_device_mem_alloc_flag_t device_flags,
                       ze_device_handle_t *device,
                       size_t alignment)
{
	int err;
	uint32_t count = 1;
	struct aml_area *out = NULL;
	struct aml_area_ze_data *data;
	ze_context_desc_t desc = {.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC,
	                          .pNext = NULL,
	                          .flags = 0};

	// Alloc area and set area fields.
	out = AML_INNER_MALLOC(struct aml_area, struct aml_area_ze_data,
	                       ze_device_handle_t);
	if (out == NULL)
		return -AML_ENOMEM;
	data = AML_INNER_MALLOC_GET_FIELD(out, 2, struct aml_area,
	                                  struct aml_area_ze_data,
	                                  ze_device_handle_t);
	out->data = (struct aml_area_data *)data;

	// Get first driver
	err = ZE(zeDriverGet(&count, &data->driver));
	if (err != AML_SUCCESS)
		goto err_with_area;

	// Create a context for this area
	err = ZE(zeContextCreate(data->driver, &desc, &data->context));
	if (err != AML_SUCCESS)
		goto err_with_area;

	// Initialize device field
	if (device == NULL) {
		// Get a default device
		if (alloc_type & AML_AREA_ZE_MMAP_DEVICE_FLAGS) {
			count = 1;
			data->device = AML_INNER_MALLOC_GET_FIELD(
			        out, 3, struct aml_area,
			        struct aml_area_ze_data, ze_device_handle_t);
			err = zeDeviceGet(data->driver, &count, data->device);
			if (err != AML_SUCCESS)
				goto err_with_context;
		}
		// No device required.
		else {
			data->device = NULL;
		}
	}
	// A device is provided then copy it.
	else {
		data->device = AML_INNER_MALLOC_GET_FIELD(
		        out, 3, struct aml_area, struct aml_area_ze_data,
		        ze_device_handle_t);
		memcpy(data->device, device, sizeof(*device));
	}

	// Initialize other fields.
	data->alignment = alignment;
	data->host_desc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
	data->host_desc.pNext = NULL;
	data->host_desc.flags = host_flags;
	data->device_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
	data->device_desc.pNext = NULL;
	data->device_desc.flags = device_flags;

	// Set op table:
	switch (alloc_type) {
	case AML_AREA_ZE_MMAP_HOST_FLAGS:
		out->ops = &aml_area_ze_ops_host;
		break;
	case AML_AREA_ZE_MMAP_DEVICE_FLAGS:
		out->ops = &aml_area_ze_ops_device;
		break;
	case AML_AREA_ZE_MMAP_SHARED_FLAGS:
		out->ops = &aml_area_ze_ops_shared;
		break;
	}

	*area = out;
	return AML_SUCCESS;
err_with_context:
	zeContextDestroy(data->context);
err_with_area:
	free(out);
	return err;
}

void aml_area_ze_destroy(struct aml_area **area)
{
	if (area != NULL && *area != NULL) {
		struct aml_area_ze_data *data =
		        (struct aml_area_ze_data *)((*area)->data);
		zeContextDestroy(data->context);
		free(*area);
		*area = NULL;
	}
}

void *aml_area_ze_mmap_host(const struct aml_area_data *area_data,
                            size_t size,
                            struct aml_area_mmap_options *options)
{
	void *ptr;
	(void)options;
	struct aml_area_ze_data *data = (struct aml_area_ze_data *)area_data;
	int err = ZE(zeMemAllocHost(data->context, &data->host_desc, size,
	                            data->alignment, &ptr));
	if (err != AML_SUCCESS) {
		aml_errno = err;
		return NULL;
	}
	return ptr;
}

int aml_area_ze_munmap(const struct aml_area_data *area_data,
                       void *ptr,
                       const size_t size)
{
	(void)size;
	struct aml_area_ze_data *data = (struct aml_area_ze_data *)area_data;
	return ZE(zeMemFree(data->context, ptr));
}

struct aml_area_ops aml_area_ze_ops_host = {
        .mmap = aml_area_ze_mmap_host,
        .munmap = aml_area_ze_munmap,
        .fprintf = NULL,
};

void *aml_area_ze_mmap_device(const struct aml_area_data *area_data,
                              size_t size,
                              struct aml_area_mmap_options *options)
{
	void *ptr;
	(void)options;
	struct aml_area_ze_data *data = (struct aml_area_ze_data *)area_data;
	int err = ZE(zeMemAllocDevice(data->context, &data->device_desc, size,
	                              data->alignment, *data->device, &ptr));
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
	struct aml_area_ze_data *data = (struct aml_area_ze_data *)area_data;
	int err = ZE(zeMemAllocShared(data->context, &data->device_desc,
	                              &data->host_desc, size, data->alignment,
	                              *data->device, &ptr));
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
