/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
*******************************************************************************/

#ifndef AML_DMA_LINUX_PAR_H
#define AML_DMA_LINUX_PAR_H 1

/*******************************************************************************
 * Linux Parallel DMA API:
 * DMA logic implemented based on general linux API, with the caller thread
 * used as the only execution thread.
 ******************************************************************************/

extern struct aml_dma_ops aml_dma_linux_par_ops;

struct aml_dma_linux_par_thread_data {
	int tid;
	pthread_t thread;
	struct aml_dma_linux_par *dma;
	struct aml_dma_request_linux_par *req;
};

struct aml_dma_request_linux_par {
	int type;
	void *dest;
	void *src;
	size_t size;
	struct aml_dma_linux_par_thread_data *thread_data;
};

struct aml_dma_linux_par_data {
	size_t nbthreads;
	struct aml_vector requests;
	pthread_mutex_t lock;
};

struct aml_dma_linux_par_ops {
	void *(*do_thread)(void *thread_data);
	int (*do_copy)(struct aml_dma_linux_par_data *data,
		       struct aml_dma_request_linux_par *request, int tid);
};

struct aml_dma_linux_par {
	struct aml_dma_linux_par_ops ops;
	struct aml_dma_linux_par_data data;
};

#define AML_DMA_LINUX_PAR_DECL(name) \
	struct aml_dma_linux_par __ ##name## _inner_data; \
	struct aml_dma name = { \
		&aml_dma_linux_par_ops, \
		(struct aml_dma_data *)&__ ## name ## _inner_data, \
	}

#define AML_DMA_LINUX_PAR_ALLOCSIZE \
	(sizeof(struct aml_dma_linux_par) + \
	 sizeof(struct aml_dma))

/**
 * Allocates and initializes a new parallel DMA.
 *
 * @param dma an address where the pointer to the newly allocated DMA structure
 * will be stored.
 * @param nbreqs the initial number of slots for asynchronous requests that are
 * in-flight (will be increased automatically if necessary).
 * @param nbthreads the number of threads to launch for each request.
 *
 * @return 0 if successful; an error code otherwise.
 **/
int aml_dma_linux_par_create(struct aml_dma **dma, size_t nbreqs,
			     size_t nbthreads);

/**
 * Initializes a new parallel DMA.
 *
 * @param dma a pointer to a dma declared with the AML_DMA_LINUX_PAR_DECL macro
 * @param nbreqs the initial number of slots for asynchronous requests that are
 * in-flight (will be increased automatically if necessary).
 * @param nbthreads the number of threads to launch for each request.
 *
 * @return 0 if successful; an error code otherwise.
 **/
int aml_dma_linux_par_init(struct aml_dma *dma, size_t nbreqs,
			   size_t nbthreads);

/**
 * Finalize a parallel DMA
 **/
void aml_dma_linux_par_fini(struct aml_dma *dma);

/**
 * Tears down a parallel DMA created with aml_dma_linux_par_create.
 * @param dma the address of a pointer to a parallel dma. Will be NULL after.
 */
void aml_dma_linux_par_destroy(struct aml_dma **dma);

#endif // AML_LINUX_DMA_LINUX_PAR_H
