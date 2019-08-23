/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
*******************************************************************************/

/**
 * @defgroup aml_area_cuda "AML Cuda Areas"
 * @brief Cuda Implementation of Areas.
 * #include <aml/area/cuda.h>
 *
 * Cuda implementation of AML areas.
 * This building block relies on Cuda implementation of
 * malloc/free to provide mmap/munmap on device memory.
 * Additional documentation of cuda runtime API can be found here:
 * @see https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html
 *
 * AML cuda areas may be created to allocate current or specific cuda devices.
 * Also allocations can be private to a single device or shared across devices.
 * Finally allocations can be backed by host memory allocation.
 *
 * @{
 **/

/**
 * Structure containing aml area hooks for cuda implementation.
 * For now there is only a single implementation of the hooks.
 * This implementation will choose between different cuda functions.
 **/
extern struct aml_area_ops aml_area_cuda_ops;

/**
 * Default cuda area with private mapping in current device.
 * Can be used out of the box with aml_area_*() functions.
 **/
extern struct aml_area aml_area_cuda;

/**
 * Allocation flags to pass to cudaMallocManaged().
 * @see cuda runtime API documentation / memory management.
 **/
enum aml_area_cuda_flags {
	AML_AREA_CUDA_ATTACH_GLOBAL,
	AML_AREA_CUDA_ATTACH_HOST,
};

/** Implementation of aml_area_data for cuda areas. **/
struct aml_area_cuda_data {
	/** allocation flags in cuda format **/
	int flags;
	/** The device id on which allocation is done. **/
	int device;
};

/**
 * \brief Cuda area creation.
 *
 * @param[out] area pointer to an uninitialized struct aml_area pointer to
 * receive the new area.
 * @param[in] device: A valid cuda device id, i.e from 0 to num_devices-1.
 * If device id is negative, then no cuda device will be selected when
 * using aml_area_cuda_mmap().
 * @param[in] flags: Allocation flags.
 *
 * @return AML_SUCCESS on success and area points to the new aml_area.
 * @return -AML_FAILURE if cuda API failed to provide the number of devices.
 * @return -AML_EINVAL if device id is greater than or equal to the number
 * of devices.
 * @return -AML_ENOMEM if space to carry area cannot be allocated.
 *
 * @see enum aml_area_cuda_flags.
 **/
int aml_area_cuda_create(struct aml_area **area,
			 const int device,
			 const enum aml_area_cuda_flags flags);

/**
 * \brief Cuda area destruction.
 *
 * Destroy (finalize and free resources) a struct aml_area created by
 * aml_area_cuda_create().
 *
 * @param[in, out] area is NULL after this call.
 **/
void aml_area_cuda_destroy(struct aml_area **area);

/**
 * \brief Cuda implementation of mmap operation for aml area.
 *
 * This function is a wrapper on cuda alloc functions.
 * It uses area settings to: select device on which to perform allocation,
 * select allocation function and set its parameters.
 * Allocations can be standalone on device, shared across multiple devices,
 * and backed with cpu memory.
 * Device selection is not thread safe and requires to set the global
 * state of cuda library. When selecting a device, allocation may succeed
 * while setting device back to original context devices may fail. In that
 * case, you need to set aml_errno to AML_SUCCESS prior to calling this
 * function in order to catch the error when return value is not NULL.
 *
 * @param[in] area_data: The structure containing cuda area settings.
 * @param[in, out] ptr: If ptr is NULL, then call cudaMallocManaged() with
 * area flags. Memory will be allocated only device side.
 * If ptr is not NULL:
 * * ptr must point to a valid memory area.
 * Device side memory will be mapped on this host side memory.
 * According to cuda runtime API documentation
 * (cudaHostRegister()), host side memory pages will be locked or allocation
 * will fail.
 * @param[in] size: The size to allocate.
 *
 * @return A cuda pointer to allocated device memory on success, NULL on
 * failure. If failure occures, aml_errno variable is set with one of the
 * following values:
 * * AML_ENOTSUP is one of the cuda calls failed with error:
 * cudaErrorInsufficientDriver, cudaErrorNoDevice.
 * * AML_EINVAL if target device id is not valid.
 * * AML_EBUSY if a specific device was requested and call to failed with error
 * cudaErrorDeviceAlreadyInUse, or if region was already mapped on device.
 * * AML_ENOMEM if memory allocation failed with error
 * cudaErrorMemoryAllocation.
 * * AML_FAILURE if one of the cuda calls resulted in error
 * cudaErrorInitializationError.
 **/
void *aml_area_cuda_mmap(const struct aml_area_data *area_data,
			 void *ptr, size_t size);

/**
 * \brief munmap hook for aml area.
 *
 * unmap memory mapped with aml_area_cuda_mmap().
 * @param[in] area_data: Ignored
 * @param[in, out] ptr: The virtual memory to unmap.
 * @param[in] size: The size of virtual memory to unmap.
 * @return -AML_EINVAL if cudaFree() returned cudaErrorInvalidValue.
 * @return AML_SUCCESS otherwise.
 **/
int
aml_area_cuda_munmap(const struct aml_area_data *area_data,
		     void *ptr, const size_t size);

/**
 * @}
 **/
