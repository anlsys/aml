/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#ifndef AML_UTILS_BACKEND_ZE_H
#define AML_UTILS_BACKEND_ZE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <level_zero/ze_api.h>

/**
 * @defgroup aml_backend_ze "AML Level Zero Utils"
 * @brief Boilerplate Code and Initialization for Level Zero Backend.
 * @code
 * #include <aml/utils/backend/ze.h>
 * @endcode
 *
 * This header can only be included if ` AML_HAVE_BACKEND_ZE == 1`.
 * @{
 **/

/**
 * Pseudo topology of ze devices and memories has provided
 * by one level zero driver.
 */
struct aml_ze_data {
	// Driver
	ze_driver_handle_t driver;
	// Number of physical devices represented by driver.
	uint32_t num_device;
	// Collection of physical devices represented by driver.
	ze_device_handle_t *device;
	// For each device, the number of subdevices.
	uint32_t *num_subdevice;
	// Collection of subdevices for each device.
	ze_device_handle_t **subdevice;
	// For each device, for each subdevice, the number of memories.
	uint32_t **num_memories;
};

/** Default pseudo topology from first available driver. */
extern struct aml_ze_data *aml_ze_default_data;

/**
 * Allocate and initialize a pseudo topology of devices.
 *
 * @param out[out]: A pointer to where to allocate result structure.
 * @param driver_num[in]: A driver number from 0 to (num_drivers-1).
 * @return -AML_EINVAL if driver_num is not a valid driver index.
 * @return -AML_ENOMEM if there was not enough memory available to
 * allocate the result structure.
 * @return -AML_ENOTSUP if the platform has no driver or device available.
 * @return -AML_FAILURE on other unexpected error.
 * @return AML_SUCCESS on success.
 */
int aml_ze_data_create(struct aml_ze_data **out, uint32_t driver_num);

/**
 * Free a pseudo topology of devices.
 *
 * @param out[out]: A pointer to a pseudo topology allocated with
 * `aml_ze_data_create()`. The pointer content is freed and set to `NULL`.
 * @return -AML_EINVAL if `out` is `NULL` or `*out` is NULL.
 * @return AML_SUCCESS on success.
 */
int aml_ze_data_destroy(struct aml_ze_data **out);

/**
 * Level zero backend initialization function.
 * This function initializes ze library and some ze specific global variables:
 * - `aml_ze_default_data`
 * - `aml_area_ze_device`
 * - `aml_area_ze_host`
 * - `aml_dma_ze_default`
 * This function should only be called once.
 *
 * @return AML_SUCCESS on success.
 * This function succeed even if ze backend is not supported at runtime.
 * In the latter case, ze specific global variable are initialized to `NULL`.
 * @return -AML_ENOTSUP if the platform has no driver or device available.
 * @return -AML_ENOMEM if there was not enough memory available to satisfy
 * this call.
 * @return -AML_FAILURE on other unexpected error.
 */
int aml_backend_ze_init(void);

/**
 * Level zero backend cleanup function.
 *
 * @return AML_SUCCESS.
 */
int aml_backend_ze_finalize(void);

/**
 * Error conversion function from level zero library error to
 * AML errors. See function definition for more details.
 */
int aml_errno_from_ze_result(ze_result_t err);

/**
 * Base context description used across the library for initializing
 * contexts
 */
extern ze_context_desc_t aml_ze_context_desc;

/**
 * @}
 **/

#ifdef __cplusplus
}
#endif
#endif // AML_UTILS_BACKEND_ZE_H
