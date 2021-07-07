/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#ifndef AML_UTILS_ZE_H
#define AML_UTILS_ZE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <level_zero/ze_api.h>

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

int aml_ze_data_create(struct aml_ze_data **out, uint32_t driver_num);
int aml_ze_data_destroy(struct aml_ze_data **out);

int aml_ze_init(void);
int aml_ze_finalize(void);

int aml_errno_from_ze_result(ze_result_t err);
#define ZE(ze_call) aml_errno_from_ze_result(ze_call)

extern struct aml_ze_data *aml_ze_default_data;
extern ze_context_desc_t aml_ze_context_desc;

/**
 * @}
 **/

#ifdef __cplusplus
}
#endif
#endif // AML_UTILS_ZE_H
