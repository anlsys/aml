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
 *
 * @{
 **/

#include "aml/higher/mapper.h"
	
#define AML_MAPPER_REPLICASET_GLOBAL (AML_MAPPER_FLAG_SPLIT | 0x80000000)
#define AML_MAPPER_REPLICASET_SHARED (AML_MAPPER_FLAG_SPLIT | 0x40000000)
#define AML_MAPPER_REPLICASET_LOCAL (AML_MAPPER_FLAG_SPLIT | 0x20000000)

struct aml_shared_replica_config {
	// The area used to allocate new pointers.
	struct aml_area *area;
	// area options.
	struct aml_area_mmap_options *area_opts;
	// The dma engine to copy from host to area.
	struct aml_dma *dma_host_dst;
	// The dma engine memcpy operator.
	aml_dma_operator memcpy_host_dst;
	// The shared pointer allocated with area. This pointer
	// is updated for each allocation during the build, thus
	// requiring replicas to be built in a bulk synchronous fashion.
	void *ptr;	
	pthread_mutex_t lock;
	pthread_barrier_t barrier;
};

void aml_replica_build_init(struct aml_shared_replica_config *out,
                           unsigned num_sharing,
                           struct aml_area *area,
                           struct aml_area_mmap_options *opts,
                           struct aml_dma *dma_host_dst,
                           aml_dma_operator memcpy_host_dst);

void aml_replica_build_fini(struct aml_shared_replica_config *out);

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
