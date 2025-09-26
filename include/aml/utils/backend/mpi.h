/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#ifndef AML_UTILS_BACKEND_MPI_H
#define AML_UTILS_BACKEND_MPI_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup aml_backend_mpi "AML MPI Utils"
 * @brief Boilerplate Code and Initialization for MPI Backend.
 * @code
 * #include <aml/utils/backend/mpi.h>
 * @endcode
 *
 * @{
 **/

/**
 * MPI backend initialization function.
 * This function should only be called once.
 * This function should not fail unless the system is out of memory.
 *
 * @return AML_SUCCESS on success.
 * @return -AML_ENOMEM on error.
 */
int aml_backend_mpi_init(void);

/**
 * linux backend initialization function.
 */
int aml_backend_mpi_finalize(void);

/**
 * @}
 **/

#ifdef __cplusplus
}
#endif
#endif // AML_UTILS_BACKEND_MPI_H
