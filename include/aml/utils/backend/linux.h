/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#ifndef AML_UTILS_LINUX_H
#define AML_UTILS_LINUX_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup aml_backend_linux "AML Linux Utils"
 * @brief Boilerplate Code and Initialization for linux Backend.
 * @code
 * #include <aml/utils/backend/linux.h>
 * @endcode
 *
 * @{
 **/

/**
 * linux backend initialization function.
 * `aml_dma_linux_parallel` and `aml_dma_linux_sequential` can only
 * be used after this function has been successfully called.
 * This function should only be called once.
 * This function should not failed unless the system is out of memory.
 *
 * @return AML_SUCCESS on success.
 * @return -AML_ENOMEM on error.
 */
int aml_backend_linux_init(void);

/**
 * linux backend initialization function.
 */
int aml_backend_linux_finalize(void);

/**
 * @}
 **/

#ifdef __cplusplus
}
#endif
#endif // AML_UTILS_LINUX_H
