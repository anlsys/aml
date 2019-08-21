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

/**
 * @defgroup aml_dma_linux_par "AML Parallel DMA"
 * @brief Parallel DMA implementation.
 *
 * DMA logic implemented based on general linux API, asynchronous execution
 * threads. This DMA implementation moves between pointers allocated with an
 * aml_area_linux.
 * @{
 **/

/**
 * Default table of dma request operations for linux
 * parallel dma.
 **/
extern struct aml_dma_ops aml_dma_linux_par_ops;

/** Request handle for clients of the DMA. **/
struct aml_dma_request_linux_par {
	/** internal request uuid, index in the request vector. **/
	int uuid;
};

/** Inside of a parallel request for linux movement. **/
struct aml_dma_linux_par_request_data {
	/**
	 * The type of dma request
	 * @see <aml.h>
	 **/
	int type;
	/** The destination pointer of the data movement **/
	struct aml_layout *dest;
	/** The source pointer of the data movement **/
	struct aml_layout *src;
	/** The dma containing sequential operations **/
	struct aml_dma_linux_par *dma;
	/** The actual thread in charge for the request progress**/
	pthread_t thread;
};

/** Inside of a parallel dma for linux movement. **/
struct aml_dma_linux_par_data {
	struct aml_vector *requests;
	pthread_mutex_t lock;
};

/** Declaration of linux parallel dma operations **/
struct aml_dma_linux_par_ops {
	void *(*do_thread)(void *data);
};

/**
 * aml_dma structure for linux based, parallel dma movement
 * Needs to be initialized with aml_dma_linux_par_create().
 * Can be passed to generic aml_dma_*() functions.
 **/
struct aml_dma_linux_par {
	struct aml_dma_linux_par_ops ops;
	struct aml_dma_linux_par_data data;
};

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
int aml_dma_linux_par_create(struct aml_dma **dma, size_t nbreqs);

/**
 * Tears down a parallel DMA created with aml_dma_linux_par_create.
 * @param dma the address of a pointer to a parallel dma. Will be NULL after.
 **/
void aml_dma_linux_par_destroy(struct aml_dma **dma);

/**
 * @}
 **/
#endif // AML_LINUX_DMA_LINUX_PAR_H
