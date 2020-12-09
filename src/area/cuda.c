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

static inline int cuda_to_aml_alloc_error(const int cuda_error)
{
	switch (cuda_error) {
	case cudaErrorInvalidValue:
		return AML_EINVAL;
	case cudaErrorMemoryAllocation:
		return AML_ENOMEM;
	case cudaErrorNotSupported:
		return AML_ENOTSUP;
	case cudaErrorInsufficientDriver:
		return AML_ENOTSUP;
	case cudaErrorNoDevice:
		return AML_ENOTSUP;
	case cudaErrorInitializationError:
		return AML_FAILURE;
	case cudaErrorHostMemoryAlreadyRegistered:
		return AML_EBUSY;
	default:
		return AML_SUCCESS;
	}
}

int aml_area_cuda_mmap_opts(void **out,
			    const size_t size,
			    const int device, const int flags, void **ptr_map)
{
	int current_device = -1;
	int error = AML_SUCCESS;
	int cuda_flags;

	// Set target device.
	if (device >= 0) {
		error = aml_set_cuda_device(device, &current_device);
		if (error != AML_SUCCESS)
			goto cuda_fail;
	}
	// Unified Memory Allocation
	if (flags & AML_AREA_CUDA_FLAG_ALLOC_UNIFIED) {
		cuda_flags = cudaMemAttachHost;

		if (flags & AML_AREA_CUDA_FLAG_ALLOC_GLOBAL)
			cuda_flags = cudaMemAttachGlobal;

		error = cudaMallocManaged(out, size, cuda_flags);
		if (error != cudaSuccess)
			goto cuda_fail;
	}
	// Mapped Allocation
	else if (flags & AML_AREA_CUDA_FLAG_ALLOC_MAPPED) {
		cuda_flags = cudaHostAllocMapped;

		if (flags & AML_AREA_CUDA_FLAG_ALLOC_GLOBAL)
			cuda_flags |= cudaHostAllocPortable;

		if (flags & AML_AREA_CUDA_FLAG_ALLOC_HOST) {
			error = cudaHostAlloc(out, size, cuda_flags);
			if (error != cudaSuccess)
				goto cuda_fail;
		} else if (*ptr_map != NULL) {
			error = cudaHostRegister(*ptr_map, size, cuda_flags);
			if (error != cudaSuccess)
				goto cuda_fail;
			*out = *ptr_map;
		} else {
			error = AML_EINVAL;
			goto fail;
		}
	}
	// Host Allocation
	else if (flags & AML_AREA_CUDA_FLAG_ALLOC_HOST) {
		cuda_flags = cudaHostAllocDefault;

		if (flags & AML_AREA_CUDA_FLAG_ALLOC_GLOBAL)
			cuda_flags |= cudaHostAllocPortable;

		error = cudaHostAlloc(out, size, cuda_flags);
		if (error != cudaSuccess)
			goto cuda_fail;
	}
	// Device Allocation
	else {
		error = cudaMalloc(out, size);
		if (error != cudaSuccess)
			goto cuda_fail;
	}

	// restore original device
	if (device >= 0 && current_device != device)
		error = aml_set_cuda_device(current_device, NULL);

	return cuda_to_aml_alloc_error(error);

cuda_fail:
	error = cuda_to_aml_alloc_error(error);
fail:
	*out = NULL;
	return error;
}

void *aml_area_cuda_mmap(const struct aml_area_data *area_data,
			 size_t size, struct aml_area_mmap_options *options)
{

	void *out;
	void *ptr_map;
	int device = -1;
	int error;
	struct aml_area_cuda_data *data;
	struct aml_area_cuda_mmap_options *opts;

	data = (struct aml_area_cuda_data *)area_data;
	opts = (struct aml_area_cuda_mmap_options *)options;

	device = (opts != NULL && opts->device > 0) ?
		opts->device : data->device;
	ptr_map = opts == NULL ? NULL : opts->ptr;

	error =
	    aml_area_cuda_mmap_opts(&out, size, device, data->flags, &ptr_map);

	if (error != AML_SUCCESS)
		aml_errno = -error;

	return out;
}

int aml_area_cuda_munmap(const struct aml_area_data *area_data,
			 void *ptr, const size_t size)
{
	(void)size;
	int flags = ((struct aml_area_cuda_data *)area_data)->flags;
	int error;

	// Unified Memory Allocation
	if (flags & AML_AREA_CUDA_FLAG_ALLOC_UNIFIED)
		error = cudaFree(ptr);
	// Mapped Allocation
	else if (flags & AML_AREA_CUDA_FLAG_ALLOC_MAPPED) {
		if (flags & AML_AREA_CUDA_FLAG_ALLOC_HOST)
			error = cudaFreeHost(ptr);
		else
			error = cudaHostUnregister(ptr);
	}
	// Host Allocation
	else if (flags & AML_AREA_CUDA_FLAG_ALLOC_HOST)
		error = cudaFreeHost(ptr);
	// Device Allocation
	else
		error = cudaFree(ptr);

	return cuda_to_aml_alloc_error(error);
}

int aml_area_cuda_fprintf(const struct aml_area_data *data,
			  FILE *stream, const char *prefix)
{
	const struct aml_area_cuda_data *d;

	/* the fields are in an order that allows us to only test those three,
	 * and threat mapped and global as special.
	 */
	static const char * const flags[] = {
		[AML_AREA_CUDA_FLAG_DEFAULT] = "default",
		[AML_AREA_CUDA_FLAG_ALLOC_HOST] = "host",
		[AML_AREA_CUDA_FLAG_ALLOC_UNIFIED] = "unified",
	};

	fprintf(stream, "%s: area-cuda: %p\n", prefix, (void *)data);
	if (data == NULL)
		return AML_SUCCESS;

	d = (const struct aml_area_cuda_data *)data;

	fprintf(stream, "%s: device: %i", prefix, d->device);
	fprintf(stream, "%s: flags: %s\n", prefix, flags[d->flags & 3]);
	fprintf(stream, "%s: mapped: %s\n", prefix,
		(d->flags & AML_AREA_CUDA_FLAG_ALLOC_MAPPED) ? "yes" : "no");
	fprintf(stream, "%s: global: %s\n", prefix,
		(d->flags & AML_AREA_CUDA_FLAG_ALLOC_GLOBAL) ? "yes" : "no");
	return AML_SUCCESS;
}



/*******************************************************************************
 * Areas Initialization
 ******************************************************************************/

int aml_area_cuda_create(struct aml_area **area,
			 const int device,
			 const int flags)
{
	struct aml_area *ret;
	struct aml_area_cuda_data *data;
	int max_devices;

	if (cudaGetDeviceCount(&max_devices) != cudaSuccess)
		return -AML_FAILURE;

	if (device >= max_devices)
		return -AML_EINVAL;

	ret = AML_INNER_MALLOC(struct aml_area,
				      struct aml_area_cuda_data);
	if (ret == NULL)
		return -AML_ENOMEM;

	data = AML_INNER_MALLOC_GET_FIELD(ret, 2, struct aml_area,
					  struct aml_area_cuda_data);

	ret->ops = &aml_area_cuda_ops;
	ret->data = (struct aml_area_data *)data;

	data->device = device < 0 ? -1 : device;
	data->flags = flags;

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
	.flags = AML_AREA_CUDA_FLAG_DEFAULT,
	.device = -1,
};

struct aml_area_cuda_data aml_area_cuda_data_unified = {
        .flags = AML_AREA_CUDA_FLAG_ALLOC_UNIFIED,
        .device = -1,
};

struct aml_area_ops aml_area_cuda_ops = {
	.mmap = aml_area_cuda_mmap,
	.munmap = aml_area_cuda_munmap,
	.fprintf = aml_area_cuda_fprintf,
};

struct aml_area aml_area_cuda = {
	.ops = &aml_area_cuda_ops,
	.data = (struct aml_area_data *)(&aml_area_cuda_data_default)
};

struct aml_area aml_area_cuda_unified = {
        .ops = &aml_area_cuda_ops,
        .data = (struct aml_area_data *)(&aml_area_cuda_data_unified)};
