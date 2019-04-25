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
	/** The tiling used for data organization in source and destination **/
	struct aml_tiling *tiling;
	/** The source pointer of the data movement **/
	void *srcptr;
	/** The identifier of the source tile **/
	int srcid;
	/** The destination pointer of the data movement **/
	void *dstptr;
	/** The identifier of the destination tile **/
	int dstid;
	/** The request used for movement **/
	struct aml_dma_request *dma_req;
};

/** Inner data of the sequential scratchpad implementation **/
struct aml_scratch_seq_data {
	/** The source area where data comes from **/
	struct aml_area *src_area;
	/** The destination area where data temporariliy goes to **/
	struct aml_area *sch_area;
	/**
	 * The data organisation.
	 * /todo why can't source and destination tiling vary?
	 **/
	struct aml_tiling *tiling;
	/** \todo What is this? **/
	size_t scratch_size;
	/** The dma engine in charge of the transfer **/
	struct aml_dma *dma;
	/** Pointer to data in scratch destination **/
	void *sch_ptr;
	/** The tilings involved in ongoing scratch requests **/
	struct aml_vector tilemap;
	/** The set of dma requests submitted to the dma to mode data  **/
	struct aml_vector requests;
	/** A lock to submit concurrent dma requests via the scratchpad **/
	pthread_mutex_t lock;
};

/** The set of operation embeded in the sequential scratchpad **/
struct aml_scratch_seq_ops {
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
	struct aml_scratch_seq_ops ops;
	/** Data embeded in the scratchpad **/
	struct aml_scratch_seq_data data;
};

/**
 * Static declaration of a sequential scratchpad.
 * Needs to be initialized with aml_scratch_seq_init()
 * @see aml_scratch_seq_init()
 **/
#define AML_SCRATCH_SEQ_DECL(name) \
	struct aml_scratch_seq __ ##name## _inner_data; \
	struct aml_scratch name = { \
		&aml_scratch_seq_ops, \
		(struct aml_scratch_data *)&__ ## name ## _inner_data, \
	}

/** Static declaration of a sequential scratchpad size. **/
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

/**
 * @}
 **/
#endif // AML_SCRATCH_SEQ_H
