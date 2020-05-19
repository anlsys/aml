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

/**************************************************************************/
/* aml_hwloc_preferred_policy                                             */
/**************************************************************************/

/**
 * aml_area_hwloc_preferred is an area that maps memory and bind it to a set
 * of numanodes of the area with the "preferred" policy.
 * That is, the first node is used to bind as much data as possible,
 * then the second node etc.
 **/
struct aml_area_hwloc_preferred_data {
	unsigned num_nodes;
	hwloc_obj_t *numanodes;
};

/**
 * Options to provide to aml_area_hwloc_preferred_mmap.
 * * block_size is the minimum block size to map on a single numanode.
 * For instance if min_size = mmap_size then the mmap operation will
 * succeed only if all mmap_size can fit on a single numanode.
 **/
struct aml_area_hwloc_options {
	size_t min_size;
};

/**
 * Allocate an area with "preferred" policy with all the available numanodes
 * ordered from smallest distance to initiator to longest distance from
 * initiator.
 * @param[out] area: A pointer to the area to allocate.
 * @param[in] initiator: A topology object from which distance to
 * NUMANODES is computed.
 * @param[in] kind: The kind of distance to use.
 * @return -AML_EINVAL if area is NULL or initiator does not have a non empty
 * cpuset.
 * @return -AML_ENOMEM if allocation failed.
 * @return -AML_ENOTSUP if the system does not support memory binding or
 * numanodes discovery (unlikely), or if the distance matrix from initiator to
 * NUMANODES could not be found.
 * @return AML_SUCCESS on success.
 * @see <hwloc/distances.h>
 **/
int aml_area_hwloc_preferred_create(struct aml_area **area,
                                    hwloc_obj_t initiator,
                                    const enum hwloc_distances_kind_e kind);

/**
 * Allocate an area with "preferred" policy with all the available numanodes
 * ordered from smallest distance to longest distance to calling thread.
 * Calling thread must be bound otherwise this call fails,
 * @param[out] area: A pointer to the area to allocate.
 * NUMANODES is computed.
 * @param[in] kind: The kind of distance to use.
 * @return -AML_EINVAL if area is NULL.
 * @return -AML_FAILURE if calling thread is not bound or
 * hwloc_get_cpubind() failed.
 * @return -AML_ENOMEM if allocation failed.
 * @return -AML_ENOTSUP if the system does not support memory binding or
 * numanodes discovery (unlikely), or if the distance matrix from initiator to
 * NUMANODES could not be found.
 * @return AML_SUCCESS on success.
 * @see <hwloc/distances.h>
 **/
int aml_area_hwloc_preferred_local_create(
        struct aml_area **area, const enum hwloc_distances_kind_e kind);

/**
 * Free memory space allocated for a aml_area_hwloc_preferred.
 * @param[in, out] area: The area to free. area is set to NULL.
 * @return AML_SUCCESS.
 **/
int aml_area_hwloc_preferred_destroy(struct aml_area **area);

/**
 * Function to map data in memory for a aml_area_hwloc_preferred.
 * Data is mapped then bound page per page to area memories in a
 * preferred order.
 * @param[in] area_data: The area data field.
 * @param[in] size: The size of memory region to map.
 * @param opts: unused.
 * @return NULL on error.
 * On error, aml_errno is set to:
 * - AML_ENOMEM if not enough memory is available.
 * @return a pointer to a mapped memory region on success.
 **/
void *aml_area_hwloc_preferred_mmap(const struct aml_area_data *area_data,
                                    size_t size,
                                    struct aml_area_mmap_options *opts);

/**
 * Unmap memory associated with a data.
 * @param[in] data: The area data field.
 * @param[in] ptr: The pointer to the memory region to unmap.
 * @param[in] size: The size of memory region to unmap.
 * @return AML_SUCCESS.
 **/
int aml_area_hwloc_preferred_munmap(const struct aml_area_data *data,
                                    void *ptr,
                                    size_t size);

extern struct aml_area_ops aml_area_hwloc_preferred_ops;

/**
 * @}
 **/

#endif // AML_AREA_HWLOC_H
