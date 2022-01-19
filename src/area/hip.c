/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include <hip/hip_runtime_api.h>

#include "aml.h"

#include "aml/area/hip.h"

/*******************************************************************************
 * Implementation
 ******************************************************************************/

static int aml_set_hip_device(const int device, int *current_device)
{
	if (current_device != NULL && device != *current_device) {
		switch (hipGetDevice(current_device)) {
		case hipErrorInsufficientDriver:
			return -AML_ENOTSUP;
		case hipErrorNoDevice:
			return -AML_ENOTSUP;
		case hipErrorInitializationError:
			return -AML_FAILURE;
		default:
			break;
		}
	}

	if (current_device != NULL && device != *current_device) {
		switch (hipSetDevice(device)) {
		case hipErrorInvalidDevice:
			return -AML_EINVAL;
		case hipErrorContextAlreadyInUse:
			return -AML_EBUSY;
		case hipErrorInsufficientDriver:
			return -AML_ENOTSUP;
		case hipErrorNoDevice:
			return -AML_ENOTSUP;
		case hipErrorInitializationError:
			return -AML_FAILURE;
		default:
			return AML_SUCCESS;
		}
	}

	return AML_SUCCESS;
}

static inline int hip_to_aml_alloc_error(const int hip_error)
{
	switch (hip_error) {
	case hipErrorInvalidValue:
		return AML_EINVAL;
	case hipErrorMemoryAllocation:
		return AML_ENOMEM;
	case hipErrorNotSupported:
		return AML_ENOTSUP;
	case hipErrorInsufficientDriver:
		return AML_ENOTSUP;
	case hipErrorNoDevice:
		return AML_ENOTSUP;
	case hipErrorInitializationError:
		return AML_FAILURE;
	case hipErrorHostMemoryAlreadyRegistered:
		return AML_EBUSY;
	default:
		return AML_SUCCESS;
	}
}

int aml_area_hip_mmap_opts(void **out,
                           const size_t size,
                           const int device,
                           const int flags,
                           void **ptr_map)
{
	int current_device = -1;
	int error = AML_SUCCESS;
	int hip_flags;

	// Set target device.
	if (device >= 0) {
		error = aml_set_hip_device(device, &current_device);
		if (error != AML_SUCCESS)
			goto hip_fail;
	}
	// Unified Memory Allocation
	if (flags & AML_AREA_HIP_FLAG_ALLOC_UNIFIED) {
		hip_flags = hipMemAttachHost;

		if (flags & AML_AREA_HIP_FLAG_ALLOC_GLOBAL)
			hip_flags = hipMemAttachGlobal;

		error = hipMallocManaged(out, size, hip_flags);
		if (error != hipSuccess)
			goto hip_fail;
	}
	// Mapped Allocation
	else if (flags & AML_AREA_HIP_FLAG_ALLOC_MAPPED) {
		hip_flags = hipHostMallocMapped;

		if (flags & AML_AREA_HIP_FLAG_ALLOC_GLOBAL)
			hip_flags |= hipHostMallocPortable;

		if (flags & AML_AREA_HIP_FLAG_ALLOC_HOST) {
			error = hipHostMalloc(out, size, hip_flags);
			if (error != hipSuccess)
				goto hip_fail;
		} else if (*ptr_map != NULL) {
			error = hipHostRegister(*ptr_map, size, hip_flags);
			if (error != hipSuccess)
				goto hip_fail;
			*out = *ptr_map;
		} else {
			error = AML_EINVAL;
			goto fail;
		}
	}
	// Host Allocation
	else if (flags & AML_AREA_HIP_FLAG_ALLOC_HOST) {
		hip_flags = hipHostMallocDefault;

		if (flags & AML_AREA_HIP_FLAG_ALLOC_GLOBAL)
			hip_flags |= hipHostMallocPortable;

		error = hipHostMalloc(out, size, hip_flags);
		if (error != hipSuccess)
			goto hip_fail;
	}
	// Device Allocation
	else {
		error = hipMalloc(out, size);
		if (error != hipSuccess)
			goto hip_fail;
	}

	// restore original device
	if (device >= 0 && current_device != device)
		error = aml_set_hip_device(current_device, NULL);

	return hip_to_aml_alloc_error(error);

hip_fail:
	error = hip_to_aml_alloc_error(error);
fail:
	*out = NULL;
	return error;
}

void *aml_area_hip_mmap(const struct aml_area_data *area_data,
                        size_t size,
                        struct aml_area_mmap_options *options)
{

	void *out;
	void *ptr_map;
	int device = -1;
	int error;
	struct aml_area_hip_data *data;
	struct aml_area_hip_mmap_options *opts;

	data = (struct aml_area_hip_data *)area_data;
	opts = (struct aml_area_hip_mmap_options *)options;

	if (opts != NULL && opts->device > 0)
		device = opts->device;
	else
		device = data->device;

	ptr_map = opts == NULL ? NULL : opts->ptr;

	error = aml_area_hip_mmap_opts(&out, size, device, data->flags,
	                               &ptr_map);

	if (error != AML_SUCCESS)
		aml_errno = -error;

	return out;
}

int aml_area_hip_munmap(const struct aml_area_data *area_data,
                        void *ptr,
                        const size_t size)
{
	(void)size;
	int flags = ((struct aml_area_hip_data *)area_data)->flags;
	int error;

	// Unified Memory Allocation
	if (flags & AML_AREA_HIP_FLAG_ALLOC_UNIFIED)
		error = hipFree(ptr);
	// Mapped Allocation
	else if (flags & AML_AREA_HIP_FLAG_ALLOC_MAPPED) {
		if (flags & AML_AREA_HIP_FLAG_ALLOC_HOST)
			error = hipHostFree(ptr);
		else
			error = hipHostUnregister(ptr);
	}
	// Host Allocation
	else if (flags & AML_AREA_HIP_FLAG_ALLOC_HOST)
		error = hipHostFree(ptr);
	// Device Allocation
	else
		error = hipFree(ptr);

	return hip_to_aml_alloc_error(error);
}

int aml_area_hip_fprintf(const struct aml_area_data *data,
                         FILE *stream,
                         const char *prefix)
{
	const struct aml_area_hip_data *d;

	/* the fields are in an order that allows us to only test those three,
	 * and threat mapped and global as special.
	 */
	static const char *const flags[] = {
	        [AML_AREA_HIP_FLAG_DEFAULT] = "default",
	        [AML_AREA_HIP_FLAG_ALLOC_HOST] = "host",
	        [AML_AREA_HIP_FLAG_ALLOC_UNIFIED] = "unified",
	};

	fprintf(stream, "%s: area-hip: %p\n", prefix, (void *)data);
	if (data == NULL)
		return AML_SUCCESS;

	d = (const struct aml_area_hip_data *)data;

	fprintf(stream, "%s: device: %i", prefix, d->device);
	fprintf(stream, "%s: flags: %s\n", prefix, flags[d->flags & 3]);
	fprintf(stream, "%s: mapped: %s\n", prefix,
	        (d->flags & AML_AREA_HIP_FLAG_ALLOC_MAPPED) ? "yes" : "no");
	fprintf(stream, "%s: global: %s\n", prefix,
	        (d->flags & AML_AREA_HIP_FLAG_ALLOC_GLOBAL) ? "yes" : "no");
	return AML_SUCCESS;
}

/*******************************************************************************
 * Areas Initialization
 ******************************************************************************/

int aml_area_hip_create(struct aml_area **area,
                        const int device,
                        const int flags)
{
	struct aml_area *ret;
	struct aml_area_hip_data *data;
	int max_devices;

	if (hipGetDeviceCount(&max_devices) != hipSuccess)
		return -AML_FAILURE;

	if (device >= max_devices)
		return -AML_EINVAL;

	ret = AML_INNER_MALLOC(struct aml_area, struct aml_area_hip_data);
	if (ret == NULL)
		return -AML_ENOMEM;

	data = AML_INNER_MALLOC_GET_FIELD(ret, 2, struct aml_area,
	                                  struct aml_area_hip_data);

	ret->ops = &aml_area_hip_ops;
	ret->data = (struct aml_area_data *)data;

	data->device = device < 0 ? -1 : device;
	data->flags = flags;

	*area = ret;
	return AML_SUCCESS;
}

void aml_area_hip_destroy(struct aml_area **area)
{
	if (*area == NULL)
		return;

	free(*area);
	*area = NULL;
}

/*******************************************************************************
 * Areas declaration
 ******************************************************************************/

struct aml_area_hip_data aml_area_hip_data_default = {
        .flags = AML_AREA_HIP_FLAG_DEFAULT,
        .device = -1,
};

struct aml_area_hip_data aml_area_hip_data_unified = {
        .flags = AML_AREA_HIP_FLAG_ALLOC_UNIFIED,
        .device = -1,
};

struct aml_area_ops aml_area_hip_ops = {
        .mmap = aml_area_hip_mmap,
        .munmap = aml_area_hip_munmap,
        .fprintf = aml_area_hip_fprintf,
};

struct aml_area aml_area_hip = {
        .ops = &aml_area_hip_ops,
        .data = (struct aml_area_data *)(&aml_area_hip_data_default)};

struct aml_area aml_area_hip_unified = {
        .ops = &aml_area_hip_ops,
        .data = (struct aml_area_data *)(&aml_area_hip_data_unified)};
