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
	void * sch_ptr;
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

/*
 * Allocates and initializes a new sequential scratchpad.
 * "scratch": an address where the pointer to the newly allocated scratchpad
 *            structure will be stored.
 * Variadic arguments:
 * - "scratch_area": an argument of type struct aml_area*; the memory area
 *                   where the scratchpad will be allocated from.
 * - "source_area": an argument of type struct aml_area*; the memory area
 *                  containing the user data structure.
 * - "dma": an argument of type struct aml_dma*; the DMA that will be used for
 *          migrating data to and from the scratchpad.
 * - "tiling": an argument of type struct aml_tiling*; the tiling to use on the
 *             user data structure and the scratchpad.
 * - "nbtiles": an argument of type size_t; number of tiles to divide the
 *              scratchpad into.
 * - "nbreqs": an argument of type size_t; the initial number of slots for
 *             asynchronous request that are in-flight (will be increased
 *             automatically if necessary).
 * Returns 0 if successful; an error code otherwise.
 */
int aml_scratch_seq_create(struct aml_scratch **scratch, ...);
/*
 * Initializes a new sequential scratchpad.  This is a varargs-variant of the
 * aml_scratch_seq_vinit() routine.
 * "scratch": an allocated scratchpad structure.
 * Variadic arguments: see aml_scratch_seq_create().
 * Returns 0 if successful; an error code otherwise.
 */
int aml_scratch_seq_init(struct aml_scratch *scratch, ...);
/*
 * Initializes a new sequential scratchpad.
 * "scratch": an allocated scratchpad structure.
 * "args": see the variadic arguments of see aml_scratch_seq_create().
 * Returns 0 if successful; an error code otherwise.
 */
int aml_scratch_seq_vinit(struct aml_scratch *scratch, va_list args);
/*
 * Tears down an initialized sequential scratchpad.
 * "scratch": an initialized scratchpad structure.
 * Returns 0 if successful; an error code otherwise.
 */
int aml_scratch_seq_destroy(struct aml_scratch *scratch);

#endif // AML_SCRATCH_SEQ_H
