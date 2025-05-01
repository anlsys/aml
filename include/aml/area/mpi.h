/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *******************************************************************************/

#ifndef AML_AREA_MPI_H
#define AML_AREA_MPI_H

#ifdef __cplusplus
extern "C" {
#endif

#include <mpi.h>

/**
 * @defgroup aml_area_mpi "AML MPI Areas"
 * @brief Implementation of Areas with MPI.
 * @code
 * #include <aml/area/mpi.h>
 * @endcode
 *
 * Implementation of Areas on top of MPI
 *  TODO
 **/

// Area Configuration Flags

/** Implementation of aml_area_data. **/
struct aml_area_mpi_data {
};

/**
 * Free the memory associated with an area allocated
 * with `aml_area_mpi_create()`
 * @param[in,out] area: A pointer to the area to free.
 */
void aml_area_mpi_destroy(struct aml_area **area);

#ifdef __cplusplus
}
#endif
#endif // AML_AREA_MPI_H
