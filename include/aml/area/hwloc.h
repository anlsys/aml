/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#ifndef AML_AREA_HWLOC_H
#define AML_AREA_HWLOC_H

#include <hwloc.h>
#include <hwloc/distances.h>

/**
 * @defgroup aml_area_hwloc "AML hwloc Areas"
 * @brief Implementation of Areas with features from hwloc library.
 *
 * Implementation of AML areas with hwloc library.
 * This building block relies on hwloc to provide mmap/munmap.
 * hwloc areas inherit hwloc portability and provide constructor
 * to customize binding of areas.
 * This building block also include a static declaration of
 * several default areas that can be used out of the box with
 * abstract area API.
 *
 * @code
 * #include <aml/area/hwloc.h>
 * @endcode
 * @see <hwloc.h>
 * @see https://www.open-mpi.org/projects/hwloc/doc/v2.0.0/
 * @{
 **/

/**************************************************************************/
/* aml_area_hwloc                                                         */
/**************************************************************************/

// Struct definition
//////////////////////

/** User data stored inside aml hwloc area. **/
struct aml_area_hwloc_data {
	/** Nodeset where to bind data **/
	hwloc_bitmap_t nodeset;
	/** Memory binding policy **/
	hwloc_membind_policy_t policy;
};

// Methods
//////////////////////

/**
 * Create an area with specific memory binding.
 * @param[out] area: A pointer where to store newly created area.
 * @param[in] nodeset: The location where to bind memory with this area.
 * @param[in] policy: The binding policy of this area.
 * @return -AML_ENOMEM if not enough memory was available to create the area.
 * @return -AML_ENOTSUP if policy is not supported.
 * @return -AML_EDOM if nodeset is out of allowed nodeset.
 * @return AML_SUCCESS.
 **/
int aml_area_hwloc_create(struct aml_area **area,
                          hwloc_bitmap_t nodeset,
                          const hwloc_membind_policy_t policy);

/**
 * Free an area obtained with aml_area_hwloc_create().
 * @param[in] area: The pointer to area to free.
 *        The content of the pointer is set to NULL.
 **/
void aml_area_hwloc_destroy(struct aml_area **area);

/**
 * Base function for mapping and binding memory.
 * It is a wrapper to hwloc_alloc_membind().
 * @param[in] data: struct aml_area_hwloc_data* initialized with a binding.
 * @param[in] size: The size of memory to query.
 * @param[in] opts: ignored.
 * @return A pointer to mapped memory region.
 * @return NULL on failure with aml_errno se to:
 *         - AML_ENOMEM if there is not enough memory available to fulfill
 *         the operation.
 *         - AML_ENOTSUP if area allocation policy is not supported.
 *         - AML_FAILURE if binding cannot be enforced.
 **/
void *aml_area_hwloc_mmap(const struct aml_area_data *data,
                          size_t size,
                          struct aml_area_mmap_options *opts);

/**
 * Free memory allocated by aml_area_hwloc_mmap().
 * It is a wrapper to hwloc_alloc_free().
 * @param[in] data: Unused.
 * @param[in] ptr: The memory region to unmap.
 * @param[in] size: The size of memory region to unmap.
 * @return AML_SUCCESS.
 **/
int aml_area_hwloc_munmap(const struct aml_area_data *data,
                          void *ptr,
                          size_t size);

/**
 * @brief Check allocation binding against area binding settings.
 *
 * Check if the data allocated with a aml_area_hwloc is bound
 * as expected to memory, i.e it is allocated on allowed NUMA nodes
 * and with the good policy.
 * @param[in] area: A aml_area_hwloc.
 * @param[in] ptr: The data allocated with same area.
 * @param[in] size: The data size in Bytes.
 * @return 1 if ptr is bound according to area settings.
 * @return 0 otherwise.
 **/
int aml_area_hwloc_check_binding(struct aml_area *area, void *ptr, size_t size);

// static declarations
//////////////////////

/** Operations of aml_area_hwloc **/
extern struct aml_area_ops aml_area_hwloc_ops;

/** Convenience default area declaration without specific binding. **/
extern struct aml_area aml_area_hwloc_default;

/**
 * Convenience area declaration with binding on every nodes of the
 * topology and interleave allocation policy.
 **/
extern struct aml_area aml_area_hwloc_interleave;

/**
 * Convenience area declaration with binding on every nodes of the
 * topology and firsttouch allocation policy.
 **/
extern struct aml_area aml_area_hwloc_firsttouch;

/**
 * @}
 **/

#endif // AML_AREA_HWLOC_H
