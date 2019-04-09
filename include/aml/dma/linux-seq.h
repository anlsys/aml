/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
*******************************************************************************/

#ifndef AML_DMA_LINUX_SEQ_H
#define AML_DMA_LINUX_SEQ_H 1

/*******************************************************************************
 * Linux Sequential DMA API:
 * DMA logic implemented based on general linux API, with the caller thread
 * used as the only execution thread.
 ******************************************************************************/

extern struct aml_dma_ops aml_dma_linux_seq_ops;

struct aml_dma_request_linux_seq {
	int type;
	void *dest;
	void *src;
	size_t size;
};

struct aml_dma_linux_seq_data {
	struct aml_vector requests;
	pthread_mutex_t lock;
};

struct aml_dma_linux_seq_ops {
	int (*do_copy)(struct aml_dma_linux_seq_data *dma,
		       struct aml_dma_request_linux_seq *req);
};

struct aml_dma_linux_seq {
	struct aml_dma_linux_seq_ops ops;
	struct aml_dma_linux_seq_data data;
};

#define AML_DMA_LINUX_SEQ_DECL(name) \
	struct aml_dma_linux_seq __ ##name## _inner_data; \
	struct aml_dma name = { \
		&aml_dma_linux_seq_ops, \
		(struct aml_dma_data *)&__ ## name ## _inner_data, \
	}

#define AML_DMA_LINUX_SEQ_ALLOCSIZE \
	(sizeof(struct aml_dma_linux_seq) + \
	 sizeof(struct aml_dma))

/**
 * Allocates and initializes a new sequential DMA.
 *
 * @param dma an address where the pointer to the newly allocated DMA structure
 * will be stored.
 * @param nbreqs the initial number of slots for asynchronous requests that are
 * in-flight (will be increased automatically if necessary).
 *
 * @return 0 if successful; an error code otherwise.
 **/
int aml_dma_linux_seq_create(struct aml_dma **dma, size_t nbreqs);

/**
 * Initializes a new sequential DMA.
 *
 * @param dma a pointer to a dma declared with the AML_DMA_LINUX_SEQ_DECL macro
 * @param nbreqs same as the create version.
 *
 * @return 0 if successful; an error code otherwise.
 **/
int aml_dma_linux_seq_init(struct aml_dma *dma, size_t nbreqs);

/**
 * Finalize a sequential DMA
 **/
void aml_dma_linux_seq_fini(struct aml_dma *dma);

/**
 * Tears down a sequential DMA created with aml_dma_linux_seq_create.
 * @param dma the address of a pointer to a sequential dma. Will be NULL after.
 */
void aml_dma_linux_seq_destroy(struct aml_dma **dma);

/* Performs a copy request.
 * "dma" the dma_linux_seq_data associated with a linux_seq dma.
 * "req" a valid linux_seq request.
 * Returns 0 if successful; an error code otherwise.
 */
int aml_dma_linux_seq_do_copy(struct aml_dma_linux_seq_data *dma,
			      struct aml_dma_request_linux_seq *req);

#endif // AML_DMA_LINUX_SEQ_H
