/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#ifndef AML_DMA_LINUX_H
#define AML_DMA_LINUX_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup aml_dma_linux "AML DMA Linux"
 * @brief AML DMA engine implementation with with a FIFO request queue and a
 * pool of threads.
 *
 * This module implements AML dma abstraction for memory movements on host.
 * DMA requests are posted to a FIFO queue. The queue is polled by a pool of
 * threads executing them.
 *
 * @code
 * #include <aml/dma/linux.h>
 * @endcode

 * @see aml_async
 * @{
 **/

//------------------------------------------------------------------------------
// User API
//------------------------------------------------------------------------------

/**
 * Create a dma engine with a custom amount of workers.
 *
 * @param[out] dma: A pointer where to allocate the dma engine.
 * @param[in] num_threads: The number of workers running the dma operations.
 * @return -AML_ENOMEM on error, exclusively caused when being out of memory.
 * @return AML_SUCCESS on success. On success, the created dma must be destroyed
 * with `aml_dma_linux_destroy()`.
 */
int aml_dma_linux_create(struct aml_dma **dma, const size_t num_threads);

/**
 * Delete a linux dma created with `aml_dma_linux_create()`.
 *
 * @param[in, out] dma: A pointer where the dma engine has been allocated.
 * The pointer content is set to NULL after deallocation.
 * @return AML_SUCCESS.
 */
int aml_dma_linux_destroy(struct aml_dma **dma);

/**
 * Pre instantiated linux dma engine.
 * The user may use this directly after a successful call to `aml_init()`.
 * This pointer is not valid anymore after `aml_finalize()` is called.
 */
extern struct aml_dma *aml_dma_linux;

//------------------------------------------------------------------------------
// Internals API
//------------------------------------------------------------------------------

/** The dma data is simply a task scheduler. */
struct aml_dma_linux_data {
	struct aml_sched *sched;
};

/** The methods table of linux dma. */
extern struct aml_dma_ops aml_dma_linux_ops;

/**
 * The task input structure sent to the scheduler
 * workers to perform the dma operator work.
 */
struct aml_dma_linux_task_in {
	// The dma operator function.
	aml_dma_operator op;
	// Destination layout of the dma operation.
	struct aml_layout *dst;
	// Source layout of the dma operation.
	struct aml_layout *src;
	// The dma operator extra arguments.
	void *op_arg;
	// The dma request associated
	struct aml_dma_linux_request *req;
};

/**
 * The work item provided to the task scheduler.
 * This function calls the operator in `input` with its arguments
 * and stored the result error code in `output`.
 *
 * @param[in] input: A pointer to `struct aml_dma_linux_task_in`.
 * @param[out] output: A pointer to an `int` where to store the result
 * of the dma operator.
 */
void aml_dma_linux_exec_request(struct aml_task_in *input,
                                struct aml_task_out *output);

/**
 * Request Flag of requests created but not returned to user that need
 * to be destroyed.
 */
#define AML_DMA_LINUX_REQUEST_FLAGS_OWNED 0x1

/**
 * Request Flag turned on when request is finished.
 */
#define AML_DMA_LINUX_REQUEST_FLAGS_DONE 0x2

/** The dma request implementation */
struct aml_dma_linux_request {
	// The scheduler task to executing the request.
	struct aml_task task;
	// The scheduler task input.
	struct aml_dma_linux_task_in task_in;
	// The scheduler task output.
	int task_out;
	// Request flags
	int flags;
};

/**
 * The linux dma `create_request()` operator implementation.
 * Creates a pointer `struct aml_dma_linux_request` stored in `req`.
 */
int aml_dma_linux_request_create(struct aml_dma_data *data,
                                 struct aml_dma_request **req,
                                 struct aml_layout *dest,
                                 struct aml_layout *src,
                                 aml_dma_operator op,
                                 void *op_arg);

/**
 * The linux dma `wait_request()` operator implementation.
 *
 * @param[in] dma: The dma engine where request has been posted.
 * @param[in] req: A pointer to a `struct aml_dma_linux_request`.
 */
int aml_dma_linux_request_wait(struct aml_dma_data *dma,
                               struct aml_dma_request **req);

/**
 * The linux dma `barrier()` operator implementation.
 * @return The first failing request error code on error.
 * Remaining requests are not waited.
 * @return AML_SUCCESS on success.
 */
int aml_dma_linux_barrier(struct aml_dma_data *dma);

/**
 * The linux dma `destroy_request()` operator implementation.
 *
 * @param[in] dma: unused.
 * @param[in] req: A pointer to a `struct aml_dma_linux_request`.
 * The pointer is set to NULL.
 */
int aml_dma_linux_request_destroy(struct aml_dma_data *dma,
                                  struct aml_dma_request **req);

/**
 * Linux DMA operator implementation:
 * Use only with `aml_dma_linux_request_create()` or higher level
 * `aml_dma_async_copy_custom()`.
 * This copy operator is compatible only with:
 * - This dma linux implementation,
 * - Dense source and destination layouts of one dimension.
 * Make a flat copy of contiguous bytes in between two layout raw pointers.
 * The size of the byte stream is computed as the product of dimensions and
 * element size.
 *
 * @param[out] dst: The destination dense layout.
 * @param[in] src: The source dense layout.
 * @param[in] arg: Unused.
 *
 * @see aml_layout_dense
 */
int aml_dma_linux_copy_1D(struct aml_layout *dst,
                          const struct aml_layout *src,
                          void *arg);

/**
 * Linux DMA operator implementation:
 * Use only with `aml_dma_linux_request_create()` or higher level
 * `aml_dma_async_copy_custom()`.
 * Make a flat copy of contiguous bytes between two raw pointers.
 * This dma operator casts input layout pointers into `void*` and
 * assumes these are contiguous set of bytes to copy from `src` to `dst`
 * in the linux `memcpy()` fashion.
 * @param[out] dst: The destination (`void*`) of the copy casted into a
 * `struct aml_layout *`.
 * @param[in] src: The source (`void*`) of the copy casted into a
 * `struct aml_layout *`.
 * @param[in] arg: The size (`size_t`) of the copy casted into a `void*`.
 * @return AML_SUCCESS
 */
int aml_memcpy_linux(struct aml_layout *dst,
                     const struct aml_layout *src,
                     void *arg);

/**
 * Linux DMA operator implementation:
 * Generic helper to copy from one layout to another.
 * @param[out] dst: The destination layout.
 * @param[in] src: The source layout.
 * @param[in] arg: Ignored.
 * @return AML_SUCCESS
 */
int aml_copy_layout_generic(struct aml_layout *dst,
                            const struct aml_layout *src,
                            void *arg);

/**
 * @}
 **/

#ifdef __cplusplus
}
#endif

#endif // AML_DMA_LINUX_H
