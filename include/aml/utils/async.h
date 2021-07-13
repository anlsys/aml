/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#ifndef AML_ASYNC_H
#define AML_ASYNC_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup aml_async "AML Asynchronous work utils"
 * @brief AML Asynchronous work utils
 *
 * This module is used internally in the library to manage asynchronous
 * optimizations.
 * In particular, it defines a task abstraction and a work queue with
 * a thread pool used by dma operations to speedup work.
 * @{
 **/

//----------------------------------------------------------------------------//
// User task abstraction (see tests/utils/test_async.c)
//----------------------------------------------------------------------------//

/** Input to an asynchronous task **/
struct aml_task_in;
/** Output from an asynchronous task **/
struct aml_task_out;
/** Task meta data **/
struct aml_task_data;
/** Function to be executed in a task**/
typedef void (*aml_task_work)(struct aml_task_in *, struct aml_task_out *);

/** Task abstraction **/
struct aml_task {
	/** Input **/
	struct aml_task_in *in;
	/** Where to store output **/
	struct aml_task_out *out;
	/** Work to do **/
	aml_task_work fn;
	/** Metadata **/
	void *data;
};

//----------------------------------------------------------------------------//
// Implementer abstraction
//----------------------------------------------------------------------------//

/** Metadata of a thread pool **/
struct aml_sched_data;

/** Methods that thread pools must implement **/
struct aml_sched_ops {
	/** Submit a task to the pool **/
	int (*submit)(struct aml_sched_data *data, struct aml_task *task);

	/** Wait for a specific task to be completed **/
	int (*wait)(struct aml_sched_data *data, struct aml_task *task);

	/** Pull the next executed task from the pool **/
	struct aml_task *(*wait_any)(struct aml_sched_data *data);
};

/** Thread pool abstraction **/
struct aml_sched {
	/** Metadata **/
	struct aml_sched_data *data;
	/** Methods **/
	struct aml_sched_ops *ops;
};

//----------------------------------------------------------------------------//
// User interface
//----------------------------------------------------------------------------//

/** Submit a task to the pool **/
int aml_sched_submit_task(struct aml_sched *pool, struct aml_task *task);

/** Wait for a specific task to be completed **/
int aml_sched_wait_task(struct aml_sched *pool, struct aml_task *task);

/** Pull the next executed task from the pool **/
struct aml_task *aml_sched_wait_any(struct aml_sched *pool);

//----------------------------------------------------------------------------//
// Simple task scheduler with a task queue and a pool of worker pthreads.
//----------------------------------------------------------------------------//

/**
 * Create a pool of threads polling work from a common FIFO queue.
 *
 * @param nt[in]: The number of threads in the pool to execute tasks in the
 * queue. If nt == 0 then progress is made from caller thread on call to
 * `aml_sched_wait_task()` and `aml_sched_wait_any()`.
 * @return An initialized task scheduler on success.
 * Created object must be destroyed with `aml_queue_sched_destroy()`.
 * @return `NULL` on error, when the system is out of memory.
 **/
struct aml_sched *aml_queue_sched_create(const size_t nt);

/**
 * Destroy a task scheduler created with `aml_queue_sched_create()`
 *
 * @param sched[in,out]: A pointer to the scheduler to destroy.
 * The pointer content is set to NULL.
 */
void aml_queue_sched_destroy(struct aml_sched **sched);

/**
 * @}
 **/

#ifdef __cplusplus
}
#endif

#endif // AML_ASYNC_H
