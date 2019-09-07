/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
*******************************************************************************/

#ifndef AML_DMA_LINUX_SPIN_H
#define AML_DMA_LINUX_SPIN_H 1

/**
 * @defgroup aml_dma_linux_spin "AML Parallel DMA"
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
extern struct aml_dma_ops aml_dma_linux_spin_ops;

/** Inside of a parallel request for linux movement. **/
struct aml_dma_request_linux_spin {
	/**
	 * The type of dma request
	 * @see <aml.h>
	 **/
	volatile int type;
	/** The destination pointer of the data movement **/
	struct aml_layout *dest;
	/** The source pointer of the data movement **/
	struct aml_layout *src;
	/** The dma containing sequential operations **/
	struct aml_dma_linux_spin *dma;
	/** The actual thread in charge for the request progress**/
	pthread_t thread;
	pthread_spinlock_t lock;
	/** operator for this request **/
	aml_dma_operator op;
	/** operator argument for this request **/
	void *op_arg;
};

/** Inside of a parallel dma for linux movement. **/
struct aml_dma_linux_spin_data {
	struct aml_dma_request_linux_spin req;
	/** default operator for this dma **/
	aml_dma_operator default_op;
	/** default operator arg for this dma **/
	void *default_op_arg;
};

/** Declaration of linux parallel dma operations **/
struct aml_dma_linux_spin_ops {
	void *(*do_thread)(void *data);
};

/**
 * aml_dma structure for linux based, parallel dma movement
 * Needs to be initialized with aml_dma_linux_spin_create().
 * Can be passed to generic aml_dma_*() functions.
 **/
struct aml_dma_linux_spin {
	struct aml_dma_linux_spin_ops ops;
	struct aml_dma_linux_spin_data data;
};

/**
 * Allocates and initializes a new parallel DMA.
 *
 * @param dma an address where the pointer to the newly allocated DMA structure
 * will be stored.
 * @param nbreqs the initial number of slots for asynchronous requests that are
 * in-flight (will be increased automatically if necessary).
 * @param nbthreads the number of threads to launch for each request.
 * @param op: default operator
 * @return 0 if successful; an error code otherwise.
 **/
int aml_dma_linux_spin_create(struct aml_dma **dma, const cpu_set_t *cpuset,
			     aml_dma_operator op, void *op_arg);

/**
 * Tears down a parallel DMA created with aml_dma_linux_spin_create.
 * @param dma the address of a pointer to a parallel dma. Will be NULL after.
 **/
void aml_dma_linux_spin_destroy(struct aml_dma **dma);

/**
 * @}
 **/
#endif // AML_LINUX_DMA_LINUX_SPIN_H
