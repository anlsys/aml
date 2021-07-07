/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#ifndef AML_UTILS_HWLOC_H
#define AML_UTILS_HWLOC_H

#ifdef __cplusplus
extern "C" {
#endif

#include <hwloc.h>

int aml_hwloc_init(void);
int aml_hwloc_finalize(void);

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

#define OBJ_DIST(dist, i, j, row_stride, col_stride)                           \
	(dist)->values[((i)->logical_index + row_stride) * (dist)->nbobjs +    \
	               col_stride + (j)->logical_index]

#define IND_DIST(dist, i, j) (dist)->values[(i) * (dist)->nbobjs + (j)]

/**
 * Create a distance matrix of topology hops between to object types.
 **/
int aml_hwloc_distance_hop_matrix(const hwloc_obj_type_t ta,
                                  const hwloc_obj_type_t tb,
                                  struct hwloc_distances_s **s);
/**
 * Get a distance matrix between an initiator type and NUMANODEs
 * This method never fails.
 * If no distance matrix exist, one is created with hops as distance.
 * If no matrix with initiator type or NUMANODE type is found, any
 * available matrix is used. The obtained matrix is reshaped to fit
 * (initiator numa) x (initiator numa) matrix.
 **/
int aml_hwloc_get_NUMA_distance(const hwloc_obj_type_t type,
                                enum hwloc_distances_kind_e kind,
                                struct hwloc_distances_s **out);

extern hwloc_topology_t aml_topology;
extern hwloc_const_bitmap_t allowed_nodeset;

/**
 * @}
 **/

#ifdef __cplusplus
}
#endif
#endif // AML_UTILS_HWLOC_H
