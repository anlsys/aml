/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#ifndef AML_AREA_MPI_H
#define AML_AREA_MPI_H

#ifdef __cplusplus
extern "C" {
#endif

#include <mpi.h>

/**
 * @defgroup aml_area_mpi "AML MPI Areas"
 * @brief MPI RDMA Implementation of AML Areas.
 *
 * This building block relies on the MPI One-sided operations to implement
 * allocations.
 *
 * @code
 * #include <aml/area/mpi.h>
 * @endcode
 * @{
 **/

/**
 * Options that can eventually be passed to mmap
 * call.
 **/
struct aml_area_mpi_mmap_options {
	/**
	 * Specify the communicator for an allocation.
	 **/
	MPI_Comm comm;
	/**
	 * Specify the info for an allocation.
	 **/
	MPI_Info info;
	/**
	 * Return variable for the underlying window.
	 **/
	MPI_Win win;
	/**
	 * Unit displacement (in bytes)
	 **/
	int disp;
};

/**
 * Contains area operations implementation
 * for the MPI area.
 **/
extern struct aml_area_ops aml_area_mpi_ops;

/**
 * Default MPI area:
 * Uses COMM_WORLD and allocates every time.
 * Can be used out-of-the-box with aml_area_*() functions.
 **/
extern struct aml_area aml_area_mpi;

/**
 * Implementation of aml_area_data for MPI areas.
 **/
struct aml_area_mpi_data {
	/** hash table keeping track of windows **/
	struct aml_area_mpi_window *windows;
};

/**
 * \brief MPI area creation.
 *
 * Allocates and initializes struct aml_area implemented by aml_area_mpi
 * operations.
 * @param[out] area pointer to an uninitialized struct aml_area pointer to
 *       receive the new area.
 * @return On success, returns 0 and fills "area" with a pointer to the new
 *       aml_area.
 * @return On failure, fills "area" with NULL and returns one of AML error
 * codes:
 * - AML_ENOMEM if there wasn't enough memory available.
 **/
int aml_area_mpi_create(struct aml_area **area);

/**
 * \brief MPI area destruction.
 *
 * Destroys (finalizes and frees resources) struct aml_area created by
 * aml_area_mpi_create().
 *
 * @param area address of an initialized struct aml_area pointer, which will be
 * reset to NULL on return from this call.
 **/
void aml_area_mpi_destroy(struct aml_area **area);

/**
 * \brief mmap block for AML area.
 *
 * This function is a wrapper around the MPI_Win_allocate call using arguments
 * set in opts.
 * @param area_data: An aml_area_mpi_data.
 * @param size: The size to allocate.
 * @param opts: See "aml_area_mpi_mmap_options".
 * @return a valid memory pointer, or NULL on failure.
 * On failure, "errno" should be checked for further information.
 **/
void *aml_area_mpi_mmap(const struct aml_area_data *area_data,
                        size_t size,
                        struct aml_area_mmap_options *opts);

/**
 * \brief munmap hook for AML area.
 *
 * Unmaps memory mapped with aml_area_mpi_mmap().
 * @param area_data: unused
 * @param ptr: The virtual memory to unmap.
 * @param size: The size of the virtual memory to unmap.
 * @return AML_SUCCESS on success, else AML_FAILURE.
 * On failure, "errno" should be checked for further information.
 **/
int aml_area_mpi_munmap(const struct aml_area_data *area_data,
                        void *ptr,
                        const size_t size);

/**
 * @}
 **/

#ifdef __cplusplus
}
#endif
#endif // AML_AREA_MPI_H
