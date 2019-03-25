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
	void *(*do_thread)(void *);
	int (*do_copy)(struct aml_dma_linux_par_data *,
		       struct aml_dma_request_linux_par *, int tid);
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
	};

#define AML_DMA_LINUX_PAR_ALLOCSIZE \
	(sizeof(struct aml_dma_linux_par) + \
	 sizeof(struct aml_dma))

/*
 * Allocates and initializes a new parallel DMA.
 * "dma": an address where the pointer to the newly allocated DMA structure
 *        will be stored.
 * Variadic arguments:
 * - "nbreqs": an argument of type size_t; the initial number of slots for
 *             asynchronous request that are in-flight (will be increased
 *             automatically if necessary).
 * - "nbthreads": an argument of type size_t; the number of threads to launch
 *                for each request.
 * Returns 0 if successful; an error code otherwise.
 */
int aml_dma_linux_par_create(struct aml_dma **, ...);
/*
 * Initializes a new parallel DMA.  This is a varargs-variant of the
 * aml_dma_linux_par_vinit() routine.
 * "dma": an allocated DMA structure.
 * Variadic arguments: see aml_dma_linux_par_create().
 * Returns 0 if successful; an error code otherwise.
 */
int aml_dma_linux_par_init(struct aml_dma *, ...);
/*
 * Initializes a new parallel DMA.
 * "dma": an allocated DMA structure.
 * "args": see the variadic arguments of aml_dma_linux_par_create().
 * Returns 0 if successful; an error code otherwise.
 */
int aml_dma_linux_par_vinit(struct aml_dma *, va_list);
/*
 * Tears down an initialized parallel DMA.
 * "dma": an initialized DMA structure.
 * Returns 0 if successful; an error code otherwise.
 */
int aml_dma_linux_par_destroy(struct aml_dma *);

#endif // AML_LINUX_DMA_LINUX_PAR_H
