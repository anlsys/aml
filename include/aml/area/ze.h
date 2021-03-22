
/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *******************************************************************************/

#ifndef AML_AREA_ZE_H
#define AML_AREA_ZE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <level_zero/ze_api.h>

/**
 * @defgroup aml_area_ze "AML Level Zero Areas"
 * @brief Implementation of Areas with Level Zero API.
 * @code
 * #include <aml/area/ze.h>
 * @endcode
 *
 * Implementation of Areas with Level Zero API.
 * This building block relies on Ze implementation of
 * host and device memory mapping to provide mmap/munmap on device memory.
 * Additional documentation of Ze memory model can be found here:
 * @see
 * https://spec.oneapi.com/level-zero/latest/core/api.html#memory
 *
 * @{
 **/

// Area Configuration Flags

/** Host mapping. */
#define AML_AREA_ZE_MMAP_HOST_FLAGS (1UL << 0)
/** Device mapping. */
#define AML_AREA_ZE_MMAP_DEVICE_FLAGS (1UL << 1)
/** Host and Device with Unified Pointer. */
#define AML_AREA_ZE_MMAP_SHARED_FLAGS                                          \
	(AML_AREA_ZE_MMAP_HOST_FLAGS | AML_AREA_ZE_MMAP_DEVICE_FLAGS)

/** Implementation of aml_area_data. **/
struct aml_area_ze_data {
	/** Handle to the backend driver */
	ze_driver_handle_t driver;
	/** Context use for memory mapping of this area. */
	ze_context_handle_t context;
	/** Flag for tuning device mapping. */
	ze_device_mem_alloc_desc_t device_desc;
	/** Flag for tuning host mapping. */
	ze_host_mem_alloc_desc_t host_desc;
	/** Alignment. */
	size_t alignment;
	/**
	 * In case of device or shared allocator, this is
	 * the device where device data is allocated.
	 */
	ze_device_handle_t *device;
};

/** Operation table for the aml_area_ze on host */
extern struct aml_area_ops aml_area_ze_ops_host;
/** Operation table for the aml_area_ze on device */
extern struct aml_area_ops aml_area_ze_ops_device;
/** Operation table for the aml_area_ze on host and device */
extern struct aml_area_ops aml_area_ze_ops_shared;

/** Default host mapper with cached allocation and 64 bytes alignment */
extern struct aml_area *aml_area_ze_host;
/** Default device mapper with cached allocation and 64 bytes alignment */
extern struct aml_area *aml_area_ze_device;

/**
 * Instanciate a new area using level zero backend.
 * This area will use the first available driver handle.
 * This area has its own context handle.
 * @param[out] area: A pointer to the area to allocate. The resulting
 * area is stored in this pointer and can be freed with free or
 * `aml_area_ze_destroy()`.
 * @param[in] alloc_type: The target memory area: host device or both:
 * + AML_AREA_ZE_MMAP_HOST_FLAGS
 * + AML_AREA_ZE_MMAP_DEVICE_FLAGS
 * + AML_AREA_ZE_MMAP_SHARED_FLAGS
 * @param[in] host_flags: Extra flag tuning host allocator behaviour.
 * @see zeMemAllocHost().
 * @param[in] device_flags: Extra flag tuning device allocator behaviour.
 * @see zeMemAllocDevice()
 * @param[in] device: If NULL and `alloc_type` specify a device or
 * shared mapping, then the first available device of the driver is
 * selected. Else, this device is copied into the area and will be used
 * to allocate on device.
 * @param[in] alignment: Alignment of mapped pointers. Must be a power of
 * two.
 * @return AML_SUCCESS on success.
 * @return -AML_ENOMEM if there was not enough memory available to
 * satisfy this call.
 */
int aml_area_ze_create(struct aml_area **area,
                       unsigned long alloc_type,
                       ze_host_mem_alloc_flag_t host_flags,
                       ze_device_mem_alloc_flag_t device_flags,
                       ze_device_handle_t *device,
                       size_t alignment);

/**
 * Free the memory associated with an area allocated
 * with `aml_area_ze_create()`
 * @param[in,out] area: A pointer to the area to free.
 */
void aml_area_ze_destroy(struct aml_area **area);

/**
 * `mmap()` method for `struct aml_area_ze_data` allocating data on host.
 * @param[in] area_data: A pointer to a valid `struct aml_area_ze_data`.
 * @param[in] size: The size of the memory region to map.
 * @param[in] options: unused.
 * @return A pointer to the mapped data on success.
 * @return If underlying call to `zeMemAllocHost` fails with a `ze_result_t`
 * the error value is translated into an AML error and stored into
 * `aml_errno` while the function will returns `NULL`.
 * Error codes are translated as followed:
 * + ZE_RESULT_ERROR_UNINITIALIZED -> AML_FAILURE
 * + ZE_RESULT_ERROR_DEVICE_LOST -> AML_FAILURE
 * + ZE_RESULT_ERROR_INVALID_NULL_HANDLE -> AML_EINVAL
 * + ZE_RESULT_ERROR_INVALID_NULL_POINTER -> AML_EINVAL
 * + ZE_RESULT_ERROR_INVALID_ENUMERATION -> AML_EINVAL
 * + ZE_RESULT_ERROR_UNSUPPORTED_SIZE -> AML_ENOTSUP
 * + ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT -> AML_ENOTSUP
 * + ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY -> AML_ENOMEM
 * + ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY  -> AML_ENOMEM
 */
void *aml_area_ze_mmap_host(const struct aml_area_data *area_data,
                            size_t size,
                            struct aml_area_mmap_options *options);

/**
 * `mmap()` method for `struct aml_area_ze_data` allocating data on device.
 * @param[in] area_data: A pointer to a valid `struct aml_area_ze_data`.
 * @param[in] size: The size of the memory region to map.
 * @param[in] options: unused.
 * @return A pointer to the mapped data on success.
 * @return If underlying call to `zeMemAllocDevice` fails with a `ze_result_t`
 * the error value is translated into an AML error and stored into
 * `aml_errno` while the function will returns `NULL`.
 * Error codes are translated as followed:
 * + ZE_RESULT_ERROR_UNINITIALIZED -> AML_FAILURE
 * + ZE_RESULT_ERROR_DEVICE_LOST -> AML_FAILURE
 * + ZE_RESULT_ERROR_INVALID_NULL_HANDLE -> AML_EINVAL
 * + ZE_RESULT_ERROR_INVALID_NULL_POINTER -> AML_EINVAL
 * + ZE_RESULT_ERROR_INVALID_ENUMERATION -> AML_EINVAL
 * + ZE_RESULT_ERROR_UNSUPPORTED_SIZE -> AML_ENOTSUP
 * + ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT -> AML_ENOTSUP
 * + ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY -> AML_ENOMEM
 * + ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY  -> AML_ENOMEM
 */
void *aml_area_ze_mmap_device(const struct aml_area_data *area_data,
                              size_t size,
                              struct aml_area_mmap_options *options);

/**
 * `mmap()` method for `struct aml_area_ze_data` aallocating data on device
 * and host.
 * @param[in] area_data: A pointer to a valid `struct aml_area_ze_data`.
 * @param[in] size: The size of the memory region to map.
 * @param[in] options: unused.
 * @return A pointer to the mapped data on success.
 * @return If underlying call to `zeMemAllocShared` fails with a `ze_result_t`
 * the error value is translated into an AML error and stored into
 * `aml_errno` while the function will returns `NULL`.
 * Error codes are translated as followed:
 * + ZE_RESULT_ERROR_UNINITIALIZED -> AML_FAILURE
 * + ZE_RESULT_ERROR_DEVICE_LOST -> AML_FAILURE
 * + ZE_RESULT_ERROR_INVALID_NULL_HANDLE -> AML_EINVAL
 * + ZE_RESULT_ERROR_INVALID_NULL_POINTER -> AML_EINVAL
 * + ZE_RESULT_ERROR_INVALID_ENUMERATION -> AML_EINVAL
 * + ZE_RESULT_ERROR_UNSUPPORTED_SIZE -> AML_ENOTSUP
 * + ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT -> AML_ENOTSUP
 * + ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY -> AML_ENOMEM
 * + ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY  -> AML_ENOMEM
 */
void *aml_area_ze_mmap_shared(const struct aml_area_data *area_data,
                              size_t size,
                              struct aml_area_mmap_options *options);

/**
 * `unmap()` method for `struct aml_area_ze_data` to unmap data
 * mapped with one of `struct aml_area_ze_data` `mmap()` methods:
 * @see aml_area_ze_mmap_host()
 * @see aml_area_ze_mmap_device()
 * @see aml_area_ze_mmap_shared()
 * @param[in] area_data: A pointer to a valid `struct aml_area_ze_data`.
 * @param[in,out] ptr: A pointer to the memory to unmap. This pointer
 * must have been obtained with one of the `struct aml_area_ze_data`
 * `mmap()` methods.
 * @param[in] size: unused.
 * @return AML_SUCCESS or a translated `ze_result_t` into an `aml_errno`.
 * Error codes are translated as followed:
 * + ZE_RESULT_ERROR_UNINITIALIZED -> AML_FAILURE
 * + ZE_RESULT_ERROR_DEVICE_LOST -> AML_FAILURE
 * + ZE_RESULT_ERROR_INVALID_NULL_HANDLE -> AML_EINVAL
 * + ZE_RESULT_ERROR_INVALID_NULL_POINTER -> AML_EINVAL
 */
int aml_area_ze_munmap(const struct aml_area_data *area_data,
                       void *ptr,
                       const size_t size);

/**
 * @}
 **/

#ifdef __cplusplus
}
#endif
#endif // AML_AREA_ZE_H
