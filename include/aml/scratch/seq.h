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

/**
 * @defgroup aml_scratch_seq "AML Sequential Scaratchpad"
 * @brief Sequential Scratchpad implementation.
 *
 * Scratchpad uses calling thread to trigger asynchronous dma movements.
 * @{
 **/

/**
 * Default table of scratchpad operations for linux
 * sequential dma.
 **/
extern struct aml_scratch_ops aml_scratch_seq_ops;

/** Inside of a sequential scratch request with linux dma. **/
struct aml_scratch_request_seq {
	/**
	 * The type of scratchpad request
	 * @see <aml.h>
	 **/
	int type;
	/** The source layout of the data movement **/
	struct aml_layout *src;
	int srcid;
	/** The destination pointer of the data movement **/
	struct aml_layout *dst;
	int dstid;
	/** The request used for movement **/
	struct aml_dma_request *dma_req;
};

/** Inner data of the sequential scratchpad implementation **/
struct aml_scratch_seq_data {
	struct aml_tiling *src_tiling;
	struct aml_tiling *scratch_tiling;
	/** The dma engine in charge of the transfer **/
	struct aml_dma *dma;
	/** Map of tiles src layouts to scratch ids **/
	struct aml_vector *tilemap;
	/** The set of dma requests submitted to the dma to mode data  **/
	struct aml_vector *requests;
	/** A lock to submit concurrent dma requests via the scratchpad **/
	pthread_mutex_t lock;
};

/** The set of operation embeded in the sequential scratchpad **/
struct aml_scratch_seq_inner_ops {
	/**
	 * Function to submit a scratchpad request.
	 * @param scratch: The scratchpad used for the request
	 * @param req: The request to execute.
	 **/
	int (*doit)(struct aml_scratch_seq_data *scratch,
		    struct aml_scratch_request_seq *req);
};

/** Sequential implementation of a scratchpad **/
struct aml_scratch_seq {
	/** Set of operations embeded in the scratchpad **/
	struct aml_scratch_seq_inner_ops ops;
	/** Data embeded in the scratchpad **/
	struct aml_scratch_seq_data data;
};

/**
 * Allocates and initializes a new sequential scratchpad.
 *
 * @param scratch an address where the pointer to the newly allocated scratchpad
 * structure will be stored.
 *
 * @param dma the DMA that will be used for migrating data to and from
 * the scratchpad.
 * @param src_tiling the tiling on the source memory
 * @param scratch_tiling the tiling to use on the scratch
 * @param nbreqs the initial number of slots for asynchronous request that
 * are in-flight (will be increased automatically if necessary).
 * @return 0 if successful; an error code otherwise.
 **/
int aml_scratch_seq_create(struct aml_scratch **scratch,
			   struct aml_dma *dma, struct aml_tiling *src_tiling,
			   struct aml_tiling *scratch_tiling, size_t nbreqs);

/**
 * Tears down an initialized sequential scratchpad.
 *
 * @param scratch an initialized scratchpad structure. NULL on return.
 */
void aml_scratch_seq_destroy(struct aml_scratch **scratch);

/**
 * @}
 **/
#endif // AML_SCRATCH_SEQ_H
