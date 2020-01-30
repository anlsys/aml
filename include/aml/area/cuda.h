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
 * @code
 * #include <aml/area/cuda.h>
 * @endcode
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
 * Default cuda area flags.
 * * Allocation on device only,
 * * Allocation visible by a single device.
 * * Allocation not mapped on host memory.
 **/
#define AML_AREA_CUDA_FLAG_DEFAULT 0

/**
 * Device allocation flag.
 * Default behaviour is allocation on device.
 * If this flag is set then allocation will
 * be on host.
 **/
#define AML_AREA_CUDA_FLAG_ALLOC_HOST (1 << 0)

/**
 * Unified memory flag.
 * If this flag is set, then allocation will create
 * a unified memory pointer usable on host and device.
 * Additionally, AML_AREA_CUDA_FLAG_ALLOC_HOST and
 * AML_AREA_CUDA_FLAG_ALLOC_MAPPED will be ignored.
 *
 * @see cudaMallocManaged()
 **/
#define	AML_AREA_CUDA_FLAG_ALLOC_UNIFIED (1 << 1)

/**
 * Mapping flag.
 * Default behaviour is allocation not mapped.
 * If set, the pointer returned by mmap function
 * will be host side memory mapped on device.
 * A pointer to device memory can then be retrieved
 * by calling cudaHostGetDevicePointer().
 * If AML_AREA_CUDA_FLAG_ALLOC_HOST is set, then
 * host side memory will be allocated. Else,
 * "ptr" field of mmap options will be used to map
 * device memory ("ptr" must not be NULL).
 *
 * @see cudaHostRegister(), cudaHostAlloc().
 **/
#define AML_AREA_CUDA_FLAG_ALLOC_MAPPED (1 << 2)

/**
 * Unified memory setting flag.
 * If AML_AREA_CUDA_FLAG_ALLOC_UNIFIED is set,
 * then this flag is looked to set
 * cudaMallocManaged() flag cudaAttachGlobal.
 * Else if AML_AREA_CUDA_FLAG_ALLOC_MAPPED is set,
 * or AML_AREA_CUDA_FLAG_ALLOC_HOST flag is set,
 * then this flag is looked to set cudaMallocHost()
 * flag cudaHostAllocPortable.
 * The default behaviour is to make allocation
 * visible from a single device. If this flag is set,
 * then allocation will be visible on all devices.
 *
 * @see cudaMallocManaged()
 **/
#define	AML_AREA_CUDA_FLAG_ALLOC_GLOBAL (1 << 3)

/**
 * Options that can eventually be passed to mmap
 * call.
 **/
struct aml_area_cuda_mmap_options {
	/**
	 * Specify a different device for one mmap call.
	 * if device < 0 use area device.
	 **/
	int device;
	/**
	 * Host memory pointer used for mapped allocations.
	 * If flag AML_AREA_CUDA_FLAG_ALLOC_MAPPED is set
	 * and ptr is NULL, ptr will be overwritten with
	 * host allocated memory and will have to be freed
	 * using cudaFreeHost().
	 **/
	void *ptr;
};

/** aml area hooks for cuda implementation. **/
extern struct aml_area_ops aml_area_cuda_ops;

/**
 * Default cuda area:
 * Allocation on device, visible by a single device,
 * and not mapped on host memory.
 **/
extern struct aml_area aml_area_cuda;

/** Implementation of aml_area_data for cuda areas. **/
struct aml_area_cuda_data {
	/** Area allocation flags. **/
	int flags;
	/**
	 * The device id on which allocation is done.
	 * If device < 0, use current device.
	 **/
	int device;
};

/**
 * \brief Cuda area creation.
 *
 * @param[out] area pointer to an uninitialized struct aml_area pointer to
 * receive the new area.
 * @param[in] device: A valid cuda device id, i.e from 0 to num_devices-1.
 * If device id is negative, then current cuda device will be used using
 * aml_area_cuda_mmap().
 * @param[in] flags: Allocation flags.
 *
 * @return AML_SUCCESS on success and area points to the new aml_area.
 * @return -AML_FAILURE if cuda API failed to provide the number of devices.
 * @return -AML_EINVAL if device id is greater than or equal to the number
 * of devices.
 * @return -AML_ENOMEM if space to carry area cannot be allocated.
 *
 * @see AML_AREA_CUDA_FLAG_*.
 **/
int aml_area_cuda_create(struct aml_area **area,
			 const int device, const int flags);

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
 * Any pointer obtained through aml_area_cuda_mmap() must be unmapped with
 * aml_area_cuda_munmap().
 *
 * Device selection is not thread safe and requires to set the global
 * state of cuda library. When selecting a device, allocation may succeed
 * while setting device back to original context devices may fail. In that
 * case, you need to set aml_errno to AML_SUCCESS prior to calling this
 * function in order to catch the error when return value is not NULL.
 *
 * @param[in] area_data: The structure containing cuda area settings.
 * @param[in] size: The size to allocate.
 * @param[in] options: A struct aml_area_cuda_mmap_options *. If > 0,
 * device will be used to select the target device.
 * If area flags AML_AREA_CUDA_FLAG_MAPPED is set and
 * AML_AREA_CUDA_FLAG_HOST is not set, then options field "ptr" must not
 * be NULL and point to a host memory that can be mapped on GPU.
 *
 * @return NULL on failure with aml errno set to the following error codes:
 * AML_ENOTSUP is one of the cuda calls failed with error:
 * cudaErrorInsufficientDriver, cudaErrorNoDevice.
 * * AML_EINVAL if target device id is not valid or provided argument are not
 * compatible.
 * * AML_EBUSY if a specific device was requested but was in already use.
 * * AML_ENOMEM if memory allocation failed with error
 * cudaErrorMemoryAllocation.
 * * AML_FAILURE if one of the cuda calls resulted in error
 * cudaErrorInitializationError.
 * @return A cuda pointer usable on device and host if area flags contains
 * AML_AREA_CUDA_FLAG_ALLOC_UNIFIED.
 * @return A pointer to host memory on which one can call
 * cudaHostGetDevicePointer() to get a pointer to mapped device memory, if
 * AML_AREA_CUDA_FLAG_ALLOC_MAPPED is set.
 * Obtained pointer must be unmapped with aml_area_cuda_munmap(). If host side
 * memory was provided as mmap option, then it still has to be freed.
 * @return A pointer to host memory if area flag AML_AREA_CUDA_FLAG_ALLOC_HOST
 * is set.
 * @return A pointer to device memory if no flag is set.
 *
 * @see AML_AREA_CUDA_FLAG_*
 **/
void *aml_area_cuda_mmap(const struct aml_area_data *area_data,
			 size_t size, struct aml_area_mmap_options *options);

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
