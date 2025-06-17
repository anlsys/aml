/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *******************************************************************************/

#ifndef __AML_HIGHER_REPLICASET_HWLOC_H__
#define __AML_HIGHER_REPLICASET_HWLOC_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <hwloc.h>

/**
 * @defgroup aml_replicaset_hwloc "AML Replicaset hwloc"
 * @brief Build a topology aware replicaset on host processor
 * using hwloc backend.
 *
 * Replicas are created using a specific performance criterion.
 * Such performance is defined relatively to a set of initiators
 * (i.e topology object containing a cpuset) and a set of
 * NUMA nodes. For instance it is possible to create a replicaset
 * such that initiators can query pointers into relative memory with
 * lowest latency or highest bandwidth.
 *
 * #include <aml/higher/replicaset/hwloc.h>
 * @see <hwloc.h>
 * @{
 **/

typedef struct aml_replicaset_hwloc_memattr_t {
	hwloc_obj_t obj; // a numa object
	hwloc_uint64_t attr; // attr value
} aml_replicaset_hwloc_memattr_t;

/**
 * Inner data implementation of replicaset for hwloc backend.
 */
struct aml_replicaset_hwloc_data {

	/** the numa node of each replica */
	aml_replicaset_hwloc_memattr_t *numas;
};

/** Public methods struct for aml_replicaset_hwloc */
extern struct aml_replicaset_ops aml_replicaset_hwloc_ops;

/**
 * Create a replicaset abstraction where initiator can query the
 * closest replica based on `kind` criterion.
 * The replicaset is not initialized, i.e the replica pointers are
 * allocated but are not initialized.
 * @param out[out]: A pointer where to allocate the replicaset.
 * @param size[in]: Size of replicas in bytes.
 * @param initiator_type[in]: The type of topology object used to
 * compute distance to NUMA nodes. A list of unique NUMA nodes is computed
 * where each initiator point to the closest NUMA Node. For each NUMA node
 * in this list, a replica pointer is created.
 * @param kind[in]: A the kind of distance in topology to use. The NUMANODEs
 * maximizing performance for the criterion will be used for allocating
 * replicas.
 * @return -AML_ENOMEM if there was not enough memory to satisfy this call.
 * @return -AML EINVAL if initiators does not contain a cpuset.
 **/
int aml_replicaset_hwloc_create(struct aml_replicaset **out,
                                const size_t size,
                                const enum aml_replicaset_attr_kind_e kind);

/**
 * Destroy hwloc replicaset data.
 * @param replicaset: The data to destroy. It is set to NULL after destruction.
 **/
void aml_replicaset_hwloc_destroy(struct aml_replicaset **replicaset);

/**
 * Copy source data to replicas. Uses memcpy. If openmp is available,
 * the copy is parallel.
 * @param replicaset[in]: The replicaset where copies are run.
 * @param data[in]: The source data used copy.
 * @return AML_SUCCESS. Arguments check is performed in the wrapper.
 **/
int aml_replicaset_hwloc_init(struct aml_replicaset *replicaset,
                              const void *data);

/**
 * Copy data from one replica to the others.
 * If openmp is available, the copy is parallel.
 * @param replicaset[in]: The replicaset where copies are run.
 * @param id[in]: The replica index used as source of the copy.
 * @return AML_SUCCESS. Arguments check is performed in the wrapper.
 **/
int aml_replicaset_hwloc_sync(struct aml_replicaset *replicaset,
                              const unsigned int id);

/**
 * Get the replica local to calling thread.
 * @param replicaset[in]: The replicaset used to retrieve a replica.
 * @return A replica pointer local to calling thread.
 * @return NULL if calling thread is not bound.
 * @return NULL if calling thread binding is larger than replicaset initiator.
 * @return NULL if no memory were available to allocate a cpuset.
 */
void *aml_replicaset_hwloc_local_replica(struct aml_replicaset *replicaset);

/**
 * @}
 **/

#ifdef __cplusplus
}
#endif
#endif // __AML_HIGHER_REPLICASET_HWLOC_H__
