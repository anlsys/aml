/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
*******************************************************************************/

#ifndef AML_SCRATCH_SEQ_H
#define AML_SCRATCH_SEQ_H 1

/*******************************************************************************
 * Sequential scratchpad API:
 * Scratchpad uses calling thread to trigger asynchronous dma movements.
 ******************************************************************************/

extern struct aml_scratch_ops aml_scratch_seq_ops;

struct aml_scratch_request_seq {
	int type;
	struct aml_tiling *tiling;
	void *srcptr;
	int srcid;
	void *dstptr;
	int dstid;
	struct aml_dma_request *dma_req;
};

struct aml_scratch_seq_data {
	struct aml_area *src_area, *sch_area;
	struct aml_tiling *tiling;
	size_t scratch_size;
	struct aml_dma *dma;
	void *sch_ptr;
	struct aml_vector tilemap;
	struct aml_vector requests;
	pthread_mutex_t lock;
};

struct aml_scratch_seq_ops {
	int (*doit)(struct aml_scratch_seq_data *scratch,
		    struct aml_scratch_request_seq *req);
};

struct aml_scratch_seq {
	struct aml_scratch_seq_ops ops;
	struct aml_scratch_seq_data data;
};

#define AML_SCRATCH_SEQ_DECL(name) \
	struct aml_scratch_seq __ ##name## _inner_data; \
	struct aml_scratch name = { \
		&aml_scratch_seq_ops, \
		(struct aml_scratch_data *)&__ ## name ## _inner_data, \
	}

#define AML_SCRATCH_SEQ_ALLOCSIZE \
	(sizeof(struct aml_scratch_seq) + \
	 sizeof(struct aml_scratch))

/**
 * Allocates and initializes a new sequential scratchpad.
 *
 * @param scratch an address where the pointer to the newly allocated scratchpad
 * structure will be stored.
 *
 * @param scratch_area the memory area where the scratchpad will be allocated.
 * @param source_area the memory area containing the user data structure.
 * @param dma the DMA that will be used for migrating data to and from
 * the scratchpad.
 * @param tiling the tiling to use on the user data structure and the scratch.
 * @param nbtiles number of tiles to divide the scratchpad into.
 * @param nbreqs the initial number of slots for asynchronous request that
 * are in-flight (will be increased automatically if necessary).
 * @return 0 if successful; an error code otherwise.
 **/
int aml_scratch_seq_create(struct aml_scratch **scratch,
			   struct aml_area *scratch_area,
			   struct aml_area *src_area,
			   struct aml_dma *dma, struct aml_tiling *tiling,
			   size_t nbtiles, size_t nbreqs);

/**
 * Initializes a new sequential scratchpad. Similar to the create.
 *
 * @param scratch a pointer to a scratch declared with AML_SCRATCH_SEQ_DECL.
 *
 * @return 0 if successful; an error code otherwise.
 **/
int aml_scratch_seq_init(struct aml_scratch *scratch,
			   struct aml_area *scratch_area,
			   struct aml_area *src_area,
			   struct aml_dma *dma, struct aml_tiling *tiling,
			   size_t nbtiles, size_t nbreqs);

/**
 * Finalize a scratchpad.
 *
 * @param scratch a pointer to a scratch initialized by aml_scratch_seq_init
 **/
void aml_scratch_seq_fini(struct aml_scratch *scratch);

/**
 * Tears down an initialized sequential scratchpad.
 *
 * @param scratch an initialized scratchpad structure. NULL on return.
 */
void aml_scratch_seq_destroy(struct aml_scratch **scratch);

#endif // AML_SCRATCH_SEQ_H
