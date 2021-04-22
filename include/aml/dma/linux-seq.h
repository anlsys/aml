/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#ifndef AML_DMA_LINUX_SEQ_H
#define AML_DMA_LINUX_SEQ_H 1

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup aml_dma_linux_seq "AML Sequential DMA"
 * @brief Sequential DMA implementation.
 *
 * DMA logic implemented based on general linux API, with the caller thread
 * used as the only execution thread. This DMA implementation moves between
 * pointers allocated with an aml_area_linux.
 * @{
 **/

/**
 * Default table of dma request operations for linux
 * sequential dma.
 **/
extern struct aml_dma_ops aml_dma_linux_seq_ops;

/**
 * Default sequential linux dma initialized at aml_init().
 **/
extern struct aml_dma *aml_dma_linux_sequential;

/** Inside of a sequential request for linux movement. **/
struct aml_dma_request_linux_seq {
	/**
	 * The type of dma request
	 * @see <aml.h>
	 **/
	int type;
	/** The destination pointer of the data movement **/
	struct aml_layout *dest;
	/** The source pointer of the data movement **/
	struct aml_layout *src;
	/** The operator being used **/
	aml_dma_operator op;
	/** Argument for operator **/
	void *op_arg;
};

/** Inner data of sequential linux aml_dma implementation **/
struct aml_dma_linux_seq_data {
	/**
	 * Queue of submitted requests.
	 * Requests may be submitted concurrently but will all
	 * be performed by a single thread.
	 **/
	struct aml_vector *requests;
	/** Lock for queuing requests concurrently **/
	pthread_mutex_t lock;
	/** default operator **/
	aml_dma_operator default_op;
	/** default op_arg **/
	void *default_op_arg;
};

/** Declaration of available linux sequential dma operations **/
struct aml_dma_linux_seq_inner_ops {
	/**
	 * Perform a sequential copy between source and destination
	 * pointers allocated with an aml_area_linux.
	 * @see aml_area
	 **/
	int (*do_copy)(struct aml_dma_linux_seq_data *dma,
		       struct aml_dma_request_linux_seq *req);
};

/**
 * aml_dma structure for linux based, sequential dma movement.
 * Needs to be initialized with aml_dma_linux_seq_create().
 * Can be passed to generic aml_dma_*() functions.
 **/
struct aml_dma_linux_seq {
	struct aml_dma_linux_seq_inner_ops ops;
	struct aml_dma_linux_seq_data data;
};


/**
 * Allocates and initializes a new sequential DMA.
 *
 * @param dma an address where the pointer to the newly allocated DMA structure
 * will be stored.
 * @param nbreqs the initial number of slots for asynchronous requests that are
 * in-flight (will be increased automatically if necessary).
 * @param op: default operator
 * @param op_arg: default argument to the operator
 *
 * @return 0 if successful; an error code otherwise.
 **/
int aml_dma_linux_seq_create(struct aml_dma **dma, size_t nbreqs,
			     aml_dma_operator op, void *op_arg);

/**
 * Tears down a sequential DMA created with aml_dma_linux_seq_create.
 * @param dma the address of a pointer to a sequential dma. Will be NULL after.
 */
void aml_dma_linux_seq_destroy(struct aml_dma **dma);

/**
 * Performs a copy request.
 * @param dma: the dma_linux_seq_data associated with a linux_seq dma.
 * @param req: a valid linux_seq request.
 * @return 0 if successful; an error code otherwise.
 **/
int aml_dma_linux_seq_do_copy(struct aml_dma_linux_seq_data *dma,
			      struct aml_dma_request_linux_seq *req);

/**
 * @}
 **/

#ifdef __cplusplus
}
#endif
#endif // AML_DMA_LINUX_SEQ_H
