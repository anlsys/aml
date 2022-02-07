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

#define AML_MAPPER_REPLICASET_GLOBAL (AML_MAPPER_FLAG_SPLIT & (1 << 31))
#define AML_MAPPER_REPLICASET_SHARED (AML_MAPPER_FLAG_SPLIT & (1 << 30))
#define AML_MAPPER_REPLICASET_LOCAL (AML_MAPPER_FLAG_SPLIT & (1 << 29))
	
struct aml_replica_build {
	struct aml_area *area;
	struct aml_area_mmap_options *area_opts;
	struct aml_dma *dma_host_dst;
	aml_dma_operator memcpy_host_dst;
	void *ptr;
	pthread_mutex_t mutex;
	pthread_barrier_t barrier;
};

int aml_replica_build_init(struct aml_replica_build *out,
                           unsigned num_sharing,
                           struct aml_area *area,
                           struct aml_area_mmap_options *opts,
                           struct aml_dma *dma_host_dst,
                           aml_dma_operator memcpy_host_dst);

int aml_mapper_replica_build(aml_mapped_ptrs *replicaset_pointers,
                             pthread_mutex_t *replicaset_mutex,
                             struct aml_mapper_creator *crtr,
                             struct aml_replica_build *local,
                             struct aml_replica_build *shared,
                             struct aml_replica_build *global);	
/**
 * @}
 **/

#ifdef __cplusplus
}
#endif
#endif // AML_MAPPER_REPLICASET_H
