/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include <pthread.h>
#include <sched.h>

#include "aml.h"

//----------------------------------------------------------------------------//
// Generic API functions.
//----------------------------------------------------------------------------//

int aml_sched_submit_task(struct aml_sched *pool, struct aml_task *task)
{
	if (pool == NULL || pool->data == NULL || pool->ops == NULL ||
	    pool->ops->submit == NULL)
		return -AML_EINVAL;

	return pool->ops->submit(pool->data, task);
}

int aml_sched_wait_task(struct aml_sched *pool, struct aml_task *task)
{
	if (pool == NULL || pool->data == NULL || pool->ops == NULL ||
	    pool->ops->wait == NULL)
		return -AML_EINVAL;

	return pool->ops->wait(pool->data, task);
}

struct aml_task *aml_sched_wait_any(struct aml_sched *pool)
{
	if (pool == NULL || pool->data == NULL || pool->ops == NULL ||
	    pool->ops->wait_any == NULL)
		return NULL;

	return pool->ops->wait_any(pool->data);
}

//----------------------------------------------------------------------------//
// Simple task scheduler with pthread workers.
//----------------------------------------------------------------------------//

/**
 * @brief Scheduler with busy wait.
 *
 * aml_queue_sched has a pool of threads never sleeping.
 * Threads will look in a work queue for new tasks and output done
 * work to a different queue.
 *
 * In order to avoid contention on work queue lock,
 * when a thread gets the lock, it is responsible for feeding other threads
 * with work.
 **/
struct aml_queue_sched {
	/** task queue **/
	struct aml_queue *work_q;
	/** Queue lock on work_q to keep threads awake **/
	pthread_mutex_t workq_lock;
	/** work done queue **/
	struct aml_queue *done_q;
	/** Passive lock on done_q **/
	pthread_mutex_t doneq_lock;
	/** Number of threads **/
	size_t nt;
	/** threads **/
	pthread_t *threads;
};

static void *aml_queue_sched_thread_fn(void *arg)
{
	int err;
	struct aml_task *task;
	struct aml_queue_sched *sched = (struct aml_queue_sched *)arg;

	// Forever until cancelled by a call to aml_queue_sched_destroy()
loop:
	pthread_testcancel();

	// Wait for work
	if (aml_queue_len(sched->work_q) == 0) {
		sched_yield();
		goto loop;
	}

	// Poll for work:
	err = pthread_mutex_trylock(&(sched->workq_lock));
	if (err == EBUSY)
		goto loop;
	if (err != 0) {
		perror("pthread_mutex_trylock");
		goto loop;
	}
	task = aml_queue_pop(sched->work_q);
	pthread_mutex_unlock(&(sched->workq_lock));

	// Run task
	if (task == NULL)
		goto loop;
	task->fn(task->in, task->out);

	// Push task to done q
	pthread_mutex_lock(&(sched->doneq_lock));
	aml_queue_push(&(sched->done_q), (void *)task);
	pthread_mutex_unlock(&(sched->doneq_lock));

	// loop until cancelation.
	goto loop;
	return NULL;
}

// Submit task.
int aml_queue_sched_submit(struct aml_sched_data *data, struct aml_task *task)
{
	struct aml_queue_sched *sched = (struct aml_queue_sched *)data;
	int err = AML_SUCCESS;

	pthread_mutex_lock(&(sched->workq_lock));
	err = aml_queue_push(&(sched->work_q), task);
	pthread_mutex_unlock(&(sched->workq_lock));
	return err;
}

// Wait for a specific task when a pool of threads is responsible for progress.i
int aml_queue_sched_wait_async(struct aml_sched_data *data,
                               struct aml_task *task)
{
	int done = 0;
	struct aml_queue_sched *sched = (struct aml_queue_sched *)data;

	while (!done) {
		while (aml_queue_len(sched->done_q) == 0)
			sched_yield();
		pthread_mutex_lock(&(sched->doneq_lock));
		done = aml_queue_take(sched->done_q, task) != NULL;
		pthread_mutex_unlock(&(sched->doneq_lock));
	}

	return AML_SUCCESS;
}

// Wait for a specific task when the calling thread is responsible for progress.
int aml_queue_sched_wait(struct aml_sched_data *data, struct aml_task *task)
{
	struct aml_task *t;
	struct aml_queue_sched *sched = (struct aml_queue_sched *)data;

loop:
	// Look into done_q for tasks and return success if found.
	pthread_mutex_lock(&(sched->doneq_lock));
	t = aml_queue_take(sched->done_q, task);
	pthread_mutex_unlock(&(sched->doneq_lock));
	if (t != NULL)
		return AML_SUCCESS;

	// Look into work_q for the task to do.
	pthread_mutex_lock(&(sched->workq_lock));
	t = aml_queue_take(sched->work_q, task);
	pthread_mutex_unlock(&(sched->workq_lock));
	if (t != NULL)
		t->fn(t->in, t->out);
	else
		goto loop;
	return AML_SUCCESS;
}

// Wait for any task when their is a pool of threads is responsible for
// progress.
struct aml_task *aml_queue_sched_wait_any_async(struct aml_sched_data *data)
{
	struct aml_task *task = NULL;
	struct aml_queue_sched *sched = (struct aml_queue_sched *)data;

	// Poll done_q until a task is here.
	while (task == NULL) {
		while (aml_queue_len(sched->done_q) == 0)
			sched_yield();
		pthread_mutex_lock(&(sched->doneq_lock));
		task = aml_queue_pop(sched->done_q);
		pthread_mutex_unlock(&(sched->doneq_lock));
	}

	return task;
}

// Wait for any task when calling thread is responsible for progress.
struct aml_task *aml_queue_sched_wait_any(struct aml_sched_data *data)
{
	struct aml_task *task = NULL;
	struct aml_queue_sched *sched = (struct aml_queue_sched *)data;

	pthread_mutex_lock(&(sched->doneq_lock));
	task = aml_queue_pop(sched->done_q);

	if (task != NULL)
		goto found_in_doneq;

	pthread_mutex_lock(&(sched->workq_lock));
	task = aml_queue_pop(sched->work_q);
	if (task != NULL)
		task->fn(task->in, task->out);

	pthread_mutex_unlock(&(sched->workq_lock));
found_in_doneq:
	pthread_mutex_unlock(&(sched->doneq_lock));
	return task;
}

struct aml_sched *aml_queue_sched_create(const size_t num_threads)
{
	size_t i;
	struct aml_sched *sched;
	struct aml_queue_sched *data;
	struct aml_sched_ops *ops;
	const size_t qlen = num_threads == 0 ? 64 : num_threads * 8;

	// Alloc struct + pthreads + tasks slots
	if (num_threads > 0)
		sched = AML_INNER_MALLOC_ARRAY(num_threads, pthread_t,
		                               struct aml_sched,
		                               struct aml_queue_sched,
		                               struct aml_sched_ops);
	else
		sched = AML_INNER_MALLOC(struct aml_sched,
		                         struct aml_queue_sched,
		                         struct aml_sched_ops);

	if (sched == NULL) {
		aml_errno = -AML_ENOMEM;
		return NULL;
	}

	// assign data field
	data = AML_INNER_MALLOC_GET_FIELD(sched, 2, struct aml_sched,
	                                  struct aml_queue_sched,
	                                  struct aml_sched_ops);
	sched->data = (struct aml_sched_data *)data;

	// assign ops field
	ops = AML_INNER_MALLOC_GET_FIELD(sched, 3, struct aml_sched,
	                                 struct aml_queue_sched,
	                                 struct aml_sched_ops);
	if (num_threads != 0) {
		ops->submit = aml_queue_sched_submit;
		ops->wait = aml_queue_sched_wait_async;
		ops->wait_any = aml_queue_sched_wait_any_async;
	} else {
		ops->submit = aml_queue_sched_submit;
		ops->wait = aml_queue_sched_wait;
		ops->wait_any = aml_queue_sched_wait_any;
	}
	sched->ops = ops;

	// Assign ready, running and threads field in data.
	if (num_threads)
		data->threads = AML_INNER_MALLOC_GET_ARRAY(
		        sched, pthread_t, struct aml_sched,
		        struct aml_queue_sched, struct aml_sched_ops);
	else
		data->threads = NULL;

	// Alloc work queue
	data->work_q = aml_queue_create(qlen);
	if (data->work_q == NULL) {
		aml_errno = -AML_ENOMEM;
		goto failure;
	}

	// Alloc done queue
	data->done_q = aml_queue_create(qlen);
	if (data->done_q == NULL) {
		aml_errno = -AML_ENOMEM;
		goto failure_with_work_queue;
	}

	data->nt = num_threads;

	// Initialize locks
	if (pthread_mutex_init(&data->workq_lock, NULL) != 0) {
		perror("pthread_mutex_init");
		aml_errno = -AML_FAILURE;
		goto failure_with_done_queue;
	}
	if (pthread_mutex_init(&data->doneq_lock, NULL) != 0) {
		perror("pthread_mutex_init");
		aml_errno = -AML_FAILURE;
		goto failure_with_mutex;
	}

	// Start threads (if any).
	for (i = 0; i < num_threads; i++) {
		if (pthread_create(data->threads + i, NULL,
		                   aml_queue_sched_thread_fn, data) != 0)
			goto failure_with_threads;
	}

	return sched;

failure_with_threads:
	aml_errno = -AML_FAILURE;
	while (i--) {
		pthread_cancel(data->threads[i]);
		pthread_join(data->threads[i], NULL);
	}
	pthread_mutex_destroy(&data->doneq_lock);
failure_with_mutex:
	pthread_mutex_destroy(&data->workq_lock);
failure_with_done_queue:
	aml_queue_destroy(data->done_q);
failure_with_work_queue:
	aml_queue_destroy(data->work_q);
failure:
	free(sched);
	return NULL;
}

void aml_queue_sched_destroy(struct aml_sched **sched)
{
	struct aml_queue_sched *data;
	struct aml_task *task;

	if (sched == NULL || *sched == NULL)
		return;

	data = (struct aml_queue_sched *)(*sched)->data;
	for (size_t i = 0; i < data->nt; i++)
		pthread_cancel(data->threads[i]);
	for (size_t i = 0; i < data->nt; i++)
		pthread_join(data->threads[i], NULL);
	pthread_mutex_destroy(&data->workq_lock);
	pthread_mutex_destroy(&data->doneq_lock);

	task = aml_queue_pop(data->work_q);
	while (task != NULL) {
		task->fn(task->in, task->out);
		task = aml_queue_pop(data->work_q);
	}

	aml_queue_destroy(data->work_q);
	aml_queue_destroy(data->done_q);

	free(*sched);
	*sched = NULL;
}
