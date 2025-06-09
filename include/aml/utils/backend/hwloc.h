/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#ifndef AML_UTILS_BACKEND_HWLOC_H
#define AML_UTILS_BACKEND_HWLOC_H

#ifdef __cplusplus
extern "C" {
#endif

#include <hwloc.h>

/**
 * @defgroup aml_backend_hwloc "AML hwloc Utils"
 * @brief Boilerplate Code and Initialization for hwloc Backend.
 * @code
 * #include <aml/utils/backend/hwloc.h>
 * @endcode
 *
 * This header can only be included if ` AML_HAVE_BACKEND_HWLOC == 1`.
 * @{
 **/

/**
 * hwloc backend initialization function.
 * This function should only be called once.
 * This function should not fail unless hwloc backend is not supported
 * at runtime.
 *
 * @return AML_SUCCESS on success.
 * @return -AML_FAILURE on error.
 */
int aml_backend_hwloc_init(void);

/**
 * hwloc backend cleanup function.
 * This function should only be called if `aml_backend_hwloc_init()` call
 * succeeded.
 *
 * @return AML_SUCCESS.
 */
int aml_backend_hwloc_finalize(void);

/**
 * Store the smallest topology object matching the cpuset where the thread
 * calling this function is bound. This function succeeds only if the
 * calling thread or process is bound to a cpuset.
 *
 * @param out[out]: A pointer to the topology object where to store
 * the result of this function.
 * @return AML_SUCCESS on success.
 * @return -AML_FAILURE if the calling thread/process is not bound to a
 * cpuset. This function may also return this error if the cpuset where
 * the caller is bound does not match any object in the topology, which is
 * unlikely.
 * @return -AML_ENOMEM if there was not enough memory available to satisfy
 * this call.
 */
int aml_hwloc_local_initiator(hwloc_obj_t *out);

// Struct for sorting (distance) and reordering (index)
struct aml_hwloc_distance {
	unsigned index;
	hwloc_uint64_t distance;
};

/**
 * Compare distance field between two `struct aml_hwloc_distance` pointers
 * with `<` operator.
 **/
int aml_hwloc_distance_lt(const void *a_ptr, const void *b_ptr);

/**
 * Allocate and fill a distance matrix of topology hops between to
 * object types.
 *
 * @param ta[in]: An object type representing objects in the rows
 * of the resulting matrix.
 * @param tb[in]: An object type representing objects in the columns
 * of the resulting matrix.
 * @param s[out]: A pointer to the space where to allocate the matrix.
 * The pointer value is updated to point to the distance matrix.
 * @return AML_SUCCESS on success.
 * @return -AML_ENOMEM if there was not enough memory available to satisfy
 * this call.
 **/
int aml_hwloc_distance_hop_matrix(const hwloc_obj_type_t ta,
                                  const hwloc_obj_type_t tb,
                                  struct hwloc_distances_s **s);
/**
 * Get a distance matrix between NUMANODEs.
 * This function can fail only if the process is out of memory.
 * If no distance matrix exists, one is created with hops as distance.
 * The obtained matrix fits (n numa) x (n numa)
 *
 * @param kind[in]: The kind distance to store in the matrix.
 * If this distance kind between NUMA nodes available, then hop distance is used instead.
 * @param s[out]: A pointer to the space where to allocate the matrix.
 * The pointer value is updated to point to the distance matrix.
 * @return AML_SUCCESS on success.
 * @return -AML_ENOMEM if there was not enough memory available to satisfy
 * this call.
 **/
int aml_hwloc_get_NUMA_distance(enum hwloc_distances_kind_e kind,
                                struct hwloc_distances_s **out);

/**
 * Library global topology.
 * The topology is valid only after `aml_backend_hwloc_init()` has been
 * called and cease to be valid after `aml_backend_hwloc_finalize()` is
 * called.
 */
extern hwloc_topology_t aml_topology;

/**
 * Usable nodeset of the library global topology.
 * The nodeset is valid only after `aml_backend_hwloc_init()` has been
 * called and cease to be valid after `aml_backend_hwloc_finalize()` is
 * called.
 */
extern hwloc_const_bitmap_t allowed_nodeset;

/**
 * @}
 **/

#ifdef __cplusplus
}
#endif
#endif // AML_UTILS_BACKEND_HWLOC_H
