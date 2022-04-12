/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#ifndef AML_MAPPER_REPLICASET_H
#define AML_MAPPER_REPLICASET_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @addtogroup aml_mapper
 * @brief Replication of hierarchical data structures.
 *
 * Replicaset is an abstraction to describe describe how to build
 * and to build a set of copies of a hierarchical structure with some
 * members shared between replicas.
 *
 * This abstraction has been designed and intended for the following use
 * case:
 * We want to copy very large data structure on a set of device memories.
 * The large data structure has to span multiple memories and is
 * only accessed for reading (there is no automatic coherency management).
 * Some smaller members of the structure are accessed in a latency sensitive
 * fashion and the application is significantly spedup if this
 * data is replicated close the compute elements.
 *
 * This abstraction provides an interface to achieve this result.
 * It defines three levels of sharing between replicas: from the least
 * shared to the most shared, `AML_MAPPER_REPLICASET_LOCAL`,
 * `AML_MAPPER_REPLICASET_SHARED`, `AML_MAPPER_REPLICASET_GLOBAL`.
 * The replicaset abstraction create a defined number of replicas of
 * the root structure to copy. It then proceed with a parallel depth first
 * walk of the structure, once for each replica with one thread per replica
 * building the structure step by step in a synchronized fashion.
 * When going deeper in the data structure, sharing flags may only increase
 * in value meaning that elements are increasingly shared.
 * This is necessary because you cannot have private structure fields if
 * the whole structure is shared.
 * When building shared elements, threads sharing elements will elect
 * a builder thread and wait on it to update their part of the structure
 * containing the shared element.
 * The choice of the areas where global, shared and local replicas go
 * is defined by the user.
 *
 * @{
 */

/**
 * Mapper flag specifying that the structure and its descendants are to be
 * shared globally, i.e only one copy will be made and it will be shared
 * among all replicas.
 */
#define AML_MAPPER_REPLICASET_GLOBAL (AML_MAPPER_FLAG_SPLIT | 0x80000000)

/**
 * Mapper flag specifying that the structure and its descendants are to be
 * shared but not necessarilly globally. The way elements are shared will
 * depend on the implementation/configuration of the replicaset builder.
 * It could be for instance sharing at a socket level where there will be
 * one replica per socket shared between all local replicas of the socket.
 * See flag `AML_MAPPER_REPLICASET_LOCAL` for local replicas.
 */
#define AML_MAPPER_REPLICASET_SHARED (AML_MAPPER_FLAG_SPLIT | 0x40000000)

/**
 * Mapper flag specifying that the structure and its descendants are
 * private, i.e the data is not shared with any other replica.
 */
#define AML_MAPPER_REPLICASET_LOCAL (AML_MAPPER_FLAG_SPLIT | 0x20000000)

// Declaration from <aml/higher/mapper/visitor.h>
struct aml_mapper_visitor;

/**
 * Data structure describing how to build replicas for a specific
 * area and a specific sharing level.
 *
 * This structure is instanciated with: `aml_replica_build_init()`, and
 * destroyed with `aml_replica_build_fini()`.
 *
 * If the sharing level is "GLOBAL", then there should be a single instance
 * of this structure.
 * If the sharing level is "LOCAL", then there should be one instance of
 * this structure per replica.
 * If the sharing level is "SHARED", then there should be as many instance
 * of this structure as areas for this shared level.
 */
struct aml_shared_replica_config {
	// The area used to allocate new pointers.
	struct aml_area *area;
	// area options.
	struct aml_area_mmap_options *area_opts;
	// The dma engine to copy from host to area.
	struct aml_dma *dma_host_dst;
	// The dma engine memcpy operator.
	aml_dma_operator memcpy_host_dst;
	// The number of replicas sharing this config.
	unsigned int num_shared;
	// The mutex for decidind which thread is going to be the thread
	// producing the shared replica.
	pthread_mutex_t producer;
	// The number of produced replica. Each thread sharing this replica can
	// consume one.
	sem_t consumed;
	// Barrier to signal all replicas have been consumed and next step of
	// replication can resume.
	sem_t produced;
	// The shared pointer allocated with area. This pointer
	// is updated for each allocation during the build, thus
	// requiring replicas to be built in a bulk synchronous fashion.
	void *ptr;
};

/**
 * Initialize one of the structures configuring the replicaset build.
 * This function should not fail.
 *
 * @param[out] out: The structure to initialize.
 * @param[in] num_sharing: The number of replicas for this shared level.
 * This is 1 for local sharing, the total number of replica for global
 * sharing, and the number of replica in a shared group for "shared".
 * @param[in] area: The area where the replica associated with this
 * configuration will be allocated.
 * @param[in] area_opts: The area options.
 * @param[in] dma_host_dst: A dma engine to perform copies from the host
 * to the target area.
 * @param[in] memcpy_host_dst: The `memcpy` operator associated with the
 * dma engine.
 */
void aml_replica_build_init(struct aml_shared_replica_config *out,
                            unsigned num_sharing,
                            struct aml_area *area,
                            struct aml_area_mmap_options *opts,
                            struct aml_dma *dma_host_dst,
                            aml_dma_operator memcpy_host_dst);

/** Cleanup resources associated with a replicaset configuration. */
void aml_replica_build_fini(struct aml_shared_replica_config *out);

/**
 * Perform the replication process of a data structure into one of the
 * replicas of the replicaset.
 *
 * This function must be called exactly once per `LOCAL` replica, each with
 * a different `thread_handle` and `local` configuration.
 *
 * `visitor`, `global`, `replicaset_pointers` and `replicaset_mutex` must
 * be the same throughout the calls.
 *
 * `local`, `shared` and `global` are the configurations used to build
 * this replica at the corresponding sharing levels.
 * `local` is specific to this replica, `global` is the same for all
 * replicas and `shared` is the shared configuration shared by replicas
 * in the same group as this replica. Their must be exactly, as many calls
 * of this function with a same `shared` configuration as the
 * number of replicas that share this configuration. So the sum of `shared`
 * replicas must match the total number of replicas.
 *
 * @param[out] thread_handle: A pointer where to store the pthread created
 * to start the replication process for one replica. The same thread handle
 * is to be waited with `pthread_join()` to terminate the process.
 * If the creation of the replica succeeded, then the `pthread_join()`
 * function returns a pointer to the replica, else it returns NULL and
 * sets aml_errno to the error value.
 * @param[in] visitor: A visitor of the structure to visit. The visitor must
 * point to the beginning of the structure to copy.
 * @param[in] local: A pointer to a configuration structure containing the
 * area and dma for allocating and copying to this replica.
 * @param[in] shared: A pointer to a shared configuration structure
 * containing the area and dma for allocating and copying the data shared
 * across replicas with the same `shared` config as this replica.
 * @param[in] global: A pointer to the single global configuration
 * structure containing the area and dma for allocating and copying the
 * data shared across all replicas.
 * @param[in] replicaset_pointers: An array of pointers shared by all
 * replicas where the allocated regions are stored. The same array of
 * pointers can be used for cleanup of the entire replicaset
 * with `aml_mapper_deepfree()`.
 * @param[in] replicaset_mutex: A mutex used to push successfully
 * allocated and copied memory regions of the entire replicaset.
 *
 * @return AML_SUCCESS on success.
 * @return -AML_FAILURE if the pthread creation failed.
 */
int aml_mapper_replica_build_start(pthread_t *thread_handle,
                                   struct aml_mapper_visitor *visitor,
                                   struct aml_shared_replica_config *local,
                                   struct aml_shared_replica_config *shared,
                                   struct aml_shared_replica_config *global,
                                   aml_mapped_ptrs *replicaset_pointers,
                                   pthread_mutex_t *replicaset_mutex);
/**
 * @}
 **/

#ifdef __cplusplus
}
#endif
#endif // AML_MAPPER_REPLICASET_H
