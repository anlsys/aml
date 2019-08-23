/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include <cuda.h>
#include <cuda_runtime.h>
#include "aml.h"
#include "aml/area/cuda.h"

/*******************************************************************************
 * Implementation
 ******************************************************************************/

static int aml_set_cuda_device(const int device, int *current_device)
{
	if (current_device != NULL && device != *current_device) {
		switch (cudaGetDevice(current_device)) {
		case cudaErrorInsufficientDriver:
			return -AML_ENOTSUP;
		case cudaErrorNoDevice:
			return -AML_ENOTSUP;
		case cudaErrorInitializationError:
			return -AML_FAILURE;
		default:
			break;
		}
	}

	if (current_device != NULL && device != *current_device) {
		switch (cudaSetDevice(device)) {
		case cudaErrorInvalidDevice:
			return -AML_EINVAL;
		case cudaErrorDeviceAlreadyInUse:
			return -AML_EBUSY;
		case cudaErrorInsufficientDriver:
			return -AML_ENOTSUP;
		case cudaErrorNoDevice:
			return -AML_ENOTSUP;
		case cudaErrorInitializationError:
			return -AML_FAILURE;
		default:
			return AML_SUCCESS;
		}
	}

	return AML_SUCCESS;
}

static inline int handle_malloc_error(const int cuda_error)
{
	switch (cuda_error) {
	case cudaErrorInvalidValue:
		aml_errno = AML_EINVAL;
		return 1;
	case cudaErrorMemoryAllocation:
		aml_errno = AML_ENOMEM;
		return 1;
	case cudaErrorNotSupported:
		aml_errno = AML_ENOTSUP;
		return 1;
	case cudaErrorInsufficientDriver:
		aml_errno = AML_ENOTSUP;
		return 1;
	case cudaErrorNoDevice:
		aml_errno = AML_ENOTSUP;
		return 1;
	case cudaErrorInitializationError:
		aml_errno = AML_FAILURE;
		return 1;
	case cudaErrorHostMemoryAlreadyRegistered:
		aml_errno = AML_EBUSY;
		return 1;
	default:
		return 0;
	}
}

void *aml_area_cuda_mmap(const struct aml_area_data *area_data,
			 void *ptr, size_t size)
{
	(void)ptr;

	int aml_error;
	int cuda_error;
	int current_device;
	void *ret;

	struct aml_area_cuda_data *data =
	    (struct aml_area_cuda_data *)area_data;

	// Set area target device.
	if (data->device >= 0) {
		aml_error = aml_set_cuda_device(data->device, &current_device);
		if (aml_error != AML_SUCCESS) {
			aml_errno = -aml_error;
			return NULL;
		}
	}

	// Actual allocation
	if (ptr == NULL)
		cuda_error = cudaMallocManaged(&ret, size, data->flags);
	else {
		// ptr is allocated cpu memory. Then we have to map it on device
		// memory.
		cuda_error =
			cudaHostRegister(ptr, size,
					 cudaHostRegisterPortable);
		if (handle_malloc_error(cuda_error))
			return NULL;
		cuda_error = cudaHostGetDevicePointer(&ret, ptr, 0);
	}

	// Attempt to restore to original device.
	// If it fails, attempt to set aml_errno.
	// However, it might be overwritten when handling allocation
	// error code..
	if (data->device >= 0 && current_device != data->device) {
		aml_error = aml_set_cuda_device(current_device, NULL);
		aml_errno = aml_error != AML_SUCCESS ? -aml_error : aml_errno;
	}
	// Handle allocation error code.
	if (handle_malloc_error(cuda_error))
		return NULL;

	return ret;
}

int aml_area_cuda_munmap(const struct aml_area_data *area_data,
			 void *ptr, const size_t size)
{
	(void) (area_data);
	(void) (size);
	int cuda_error = cudaHostUnregister(ptr);

	if (cuda_error == cudaErrorHostMemoryNotRegistered ||
	    cuda_error == cudaErrorInvalidValue){
		cuda_error = cudaFree(ptr);
	}

	switch (cuda_error) {
	case cudaErrorInvalidValue:
		return -AML_EINVAL;
	case cudaSuccess:
		return AML_SUCCESS;
	default:
		printf("cudaError: %s\n", cudaGetErrorString(cuda_error));
		return -AML_FAILURE;
	}
}

/*******************************************************************************
 * Areas Initialization
 ******************************************************************************/

int aml_area_cuda_create(struct aml_area **area,
			 const int device,
			 const enum aml_area_cuda_flags flags)
{
	struct aml_area *ret;
	struct aml_area_cuda_data *data;
	int max_devices;

	if (cudaGetDeviceCount(&max_devices) != cudaSuccess)
		return -AML_FAILURE;

	ret = AML_INNER_MALLOC_2(struct aml_area, struct aml_area_cuda_data);
	if (ret == NULL)
		return -AML_ENOMEM;

	data = AML_INNER_MALLOC_NEXTPTR(ret, struct aml_area,
					struct aml_area_cuda_data);

	ret->ops = &aml_area_cuda_ops;
	ret->data = (struct aml_area_data *)data;
	switch (flags) {
	case AML_AREA_CUDA_ATTACH_GLOBAL:
		data->flags = cudaMemAttachGlobal;
		break;
	case AML_AREA_CUDA_ATTACH_HOST:
		data->flags = cudaMemAttachHost;
		break;
	default:
		data->flags = cudaMemAttachHost;
		break;
	}
	data->device = device < 0 || device >= max_devices ? -1 : device;

	*area = ret;
	return AML_SUCCESS;
}

void aml_area_cuda_destroy(struct aml_area **area)
{
	if (*area == NULL)
		return;

	free(*area);
	*area = NULL;
}

/*******************************************************************************
 * Areas declaration
 ******************************************************************************/

struct aml_area_cuda_data aml_area_cuda_data_default = {
	.flags = cudaMemAttachHost,
	.device = -1,
};

struct aml_area_ops aml_area_cuda_ops = {
	.mmap = aml_area_cuda_mmap,
	.munmap = aml_area_cuda_munmap
};

struct aml_area aml_area_cuda = {
	.ops = &aml_area_cuda_ops,
	.data = (struct aml_area_data *)(&aml_area_cuda_data_default)
};
