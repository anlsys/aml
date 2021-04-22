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
typedef struct aml_task_out *(*aml_task_work) (struct aml_task_in *);

/** Task abstraction **/
struct aml_task {
	/** Input **/
	struct aml_task_in *in;
	/** Where to store output **/
	struct aml_task_out *out;
	/** Work to do **/
	aml_task_work fn;
	/** Metadata **/
	struct aml_task_data *data;
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
// Simple task scheduler with pthread worker.
//----------------------------------------------------------------------------//

/**
 * Create an active pool of "nt" threads" to run asynchronously tasks queued
 * in a FIFO queue.
 * If nt == 0 then progress is made
 * from caller thread on aml_sched_wait_task() and aml_sched_wait_any().
 **/
struct aml_sched *aml_active_sched_create(const size_t nt);

/** Destroy an active thread pool and set it to NULL **/
void aml_active_sched_destroy(struct aml_sched **sched);

/** Get the number of tasks pushed to the scheduler and not yet pulled out. **/
int aml_active_sched_num_tasks(struct aml_sched *sched);

/**
 * @}
 **/

#ifdef __cplusplus
}
#endif

#endif //AML_ASYNC_H
