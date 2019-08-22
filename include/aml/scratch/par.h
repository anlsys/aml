/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
*******************************************************************************/

#ifndef AML_SCRATCH_PAR_H
#define AML_SCRATCH_PAR_H 1

/**
 * @defgroup aml_scratch_par "AML Parallel Scaratchpad"
 * @brief Parallel Scratchpad implementation.
 *
 * Scratchpad creates one thread to trigger synchronous dma movements.
 * @{
 **/

/**
 * Default table of scratchpad operations for linux
 * parallel dma.
 **/
extern struct aml_scratch_ops aml_scratch_par_ops;

/** Inside of a parallel scratch request with linux dma. **/
struct aml_scratch_request_par {
	/**
	 * The type of scratchpad request
	 * @see <aml.h>
	 **/
	int type;
	/** The source pointer of the data movement **/
	struct aml_layout *src;
	/** The tile identifier in source pointer **/
	int srcid;
	/** The destination pointer of the data movement **/
	struct aml_layout *dst;
	/** The tile identifier in destination pointer **/
	int dstid;
	/** The scratchpad handling this request **/
	struct aml_scratch_par *scratch;
	/** The thread in charge of scratch request progress **/
	pthread_t thread;
};

/**
 * Inner data of the parallel scratchpad implementation
 * \todo This is the same as struct aml_scratch_seq_data. Could be factorized
 **/
struct aml_scratch_par_data {
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
	struct aml_vector *tilemap;
	/** The set of dma requests submitted to the dma to mode data  **/
	struct aml_vector *requests;
	/** A lock to submit concurrent dma requests via the scratchpad **/
	pthread_mutex_t lock;
};

/** The set of operation embeded in the parallel scratchpad **/
struct aml_scratch_par_ops {
	/**
	 * Function to submit asynchronously scratchpad request.
	 * @param data: Argument of the thread starting the request,
	 *              i.e a struct aml_scratch_request_par.
	 * @return Unspecified value.
	 **/
	void *(*do_thread)(void *data);
};

/** Parallel implementation of a scratchpad **/
struct aml_scratch_par {
	/** Set of operations embeded in the scratchpad **/
	struct aml_scratch_par_ops ops;
	/** Data embeded in the scratchpad **/
	struct aml_scratch_par_data data;
};

/**
 * Allocates and initializes a new parallel scratchpad.
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
int aml_scratch_par_create(struct aml_scratch **scratch,
			   struct aml_area *scratch_area,
			   struct aml_area *src_area,
			   struct aml_dma *dma, struct aml_tiling *tiling,
			   size_t nbtiles, size_t nbreqs);

/**
 * Tears down an initialized parallel scratchpad.
 *
 * @param scratch an initialized scratchpad structure. NULL on return.
 */
void aml_scratch_par_destroy(struct aml_scratch **scratch);

#endif // AML_SCRATCH_PAR_H
