
/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
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

/** Device mapping. */
#define AML_AREA_ZE_MMAP_DEVICE_FLAGS (1 << 0)
/** Host and Device with Unified Pointer. */
#define AML_AREA_ZE_MMAP_SHARED_FLAGS (1 << 1)

/** Implementation of aml_area_data. **/
struct aml_area_ze_data {
	/** Context use for memory mapping of this area. */
	ze_context_handle_t context;
	/** Alignment. */
	size_t alignment;
	union ze_area_desc {
		struct ze_area_device_desc {
			/** Flag for tuning device mapping. */
			ze_device_mem_alloc_desc_t device_desc;
			/** device used for allocation. */
			ze_device_handle_t device;
		} device;
		struct ze_area_host_desc {
			/** Flag for tuning device mapping. */
			ze_host_mem_alloc_desc_t host_desc;
		} host;
	} desc;
};

/** Operation table for the aml_area_ze on device */
extern struct aml_area_ops aml_area_ze_ops_device;
/** Operation table for the aml_area_ze on host and device */
extern struct aml_area_ops aml_area_ze_ops_shared;
/** Operation table for the aml_area_ze on host */
extern struct aml_area_ops aml_area_ze_ops_host;

/**
 * Default device mapper with cached allocation and 64 bytes alignment
 * The driver used to obtain devices is the first returned driver.
 * @see zeDeviceGet()
 */
extern struct aml_area *aml_area_ze_device;

/**
 * Default hist mapper with cached allocation and 64 bytes alignment.
 * The driver used to obtain devices is the first returned driver.
 * @see zeDeviceGet()
 */
extern struct aml_area *aml_area_ze_host;

/**
 * Instanciate a new area for allocating device memory using level zero backend.
 * This area will use the first available driver handle.
 * This area has its own context handle.
 * @param[out] area: A pointer to the area to allocate. The resulting
 * area is stored in this pointer and can be freed with free or
 * `aml_area_ze_destroy()`.
 * @param[in] device: The target device where data is to be allocated.
 * @param[in] ordinal: "ordinal of the device’s local memory to allocate from."
 * @param[in] device_flags: Extra flag tuning device allocator behaviour.
 * @see zeMemAllocDevice()
 * @param[in] alignment: Alignment of mapped pointers. Must be a power of
 * two.
 * @param[in] flags: The allocation type among:
 * + AML_AREA_ZE_MMAP_DEVICE_FLAGS
 * + AML_AREA_ZE_MMAP_SHARED_FLAGS
 * @return AML_SUCCESS on success.
 * @return -AML_ENOMEM if there was not enough memory available to
 * satisfy this call.
 * @return A translated ze_result_t into AML error code if calls to ze backends
 * failed: getting drivers or context creation.
 */
int aml_area_ze_device_create(struct aml_area **area,
                              ze_device_handle_t device,
                              uint32_t ordinal,
                              ze_device_mem_alloc_flag_t device_flags,
                              size_t alignment,
                              int flags);

/**
 * Instanciate a new area for allocating host memory using level zero backend.
 * This area will use the first available driver handle.
 * This area has its own context handle.
 * @param[out] area: A pointer to the area to allocate. The resulting
 * area is stored in this pointer and can be freed with free or
 * `aml_area_ze_destroy()`.
 * @param[in] host_flags: Extra flag tuning device allocator behaviour.
 * @see zeMemAllocHost()
 * @param[in] alignment: Alignment of mapped pointers. Must be a power of
 * two.
 * @return AML_SUCCESS on success.
 * @return -AML_ENOMEM if there was not enough memory available to
 * satisfy this call.
 * @return A translated ze_result_t into AML error code if calls to ze backends
 * failed: getting drivers or context creation.
 */
int aml_area_ze_host_create(struct aml_area **area,
                            ze_host_mem_alloc_flag_t host_flags,
                            size_t alignment);

/**
 * Free the memory associated with an area allocated
 * with `aml_area_ze_create()`
 * @param[in,out] area: A pointer to the area to free.
 */
void aml_area_ze_destroy(struct aml_area **area);

/**
 * `mmap()` method for `struct aml_area_ze_data` allocating data on device.
 * @param[in] area_data: A pointer to a valid `struct aml_area_ze_data`.
 * @param[in] size: The size of the memory region to map.
 * @param[in] options: unused.
 * @return A pointer to the mapped data on success.
 * @return If underlying call to `zeMemAllocDevice()` fails with a `ze_result_t`
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
 * @return If underlying call to `zeMemAllocShared()` fails with a `ze_result_t`
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
 * `mmap()` method for `struct aml_area_ze_data` aallocating data on host.
 * @param[in] area_data: A pointer to a valid `struct aml_area_ze_data`.
 * @param[in] size: The size of the memory region to map.
 * @param[in] options: unused.
 * @return A pointer to the mapped data on success.
 * @return If underlying call to `zeMemAllocHost()` fails with a `ze_result_t`
 * the error value is translated into an AML error and stored into
 * `aml_errno` while the function will returns `NULL`.
 */
void *aml_area_ze_mmap_host(const struct aml_area_data *area_data,
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
