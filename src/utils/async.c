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
	/** Weather thread is processing work **/
	struct aml_task **in_progress;
	/** Lock guarding `in_progress` updates **/
	pthread_mutex_t in_progress_lock;
	/** Counter to set threads id from 0 to (nt-1) **/
	int tid;
};

static void *aml_queue_sched_thread_fn(void *arg)
{
	int err;
	struct aml_task *task;
	struct aml_queue_sched *sched = (struct aml_queue_sched *)arg;
	int tid;

	// Set this thread id.
	pthread_mutex_lock(&(sched->workq_lock));
	tid = sched->tid++;
	pthread_mutex_unlock(&(sched->workq_lock));

	// Forever until cancelled by a call to aml_queue_sched_destroy()
loop:
	pthread_testcancel();

	// Poll for work:
	err = pthread_mutex_trylock(&(sched->workq_lock));
	if (err == EBUSY) {
		sched_yield();
		goto loop;
	}
	if (err != 0) {
		perror("pthread_mutex_trylock");
		goto loop;
	}

	// Wait for work
	if (aml_queue_len(sched->work_q) == 0) {
		pthread_mutex_unlock(&(sched->workq_lock));
		sched_yield();
		goto loop;
	}

	task = aml_queue_pop(sched->work_q);
	if (task != NULL) {
		pthread_mutex_lock(&(sched->in_progress_lock));
		sched->in_progress[tid] = task;
		pthread_mutex_unlock(&(sched->in_progress_lock));
	}
	pthread_mutex_unlock(&(sched->workq_lock));

	// Run task
	if (task == NULL)
		goto loop;
	task->fn(task->in, task->out);

	// Push task to done q
	pthread_mutex_lock(&(sched->doneq_lock));
	pthread_mutex_lock(&(sched->in_progress_lock));
	aml_queue_push(sched->done_q, (void *)task);
	pthread_mutex_unlock(&(sched->doneq_lock));
	sched->in_progress[tid] = NULL;
	pthread_mutex_unlock(&(sched->in_progress_lock));

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
	err = aml_queue_push(sched->work_q, task);
	pthread_mutex_unlock(&(sched->workq_lock));
	return err;
}

static int comp_tasks(const void *a, const void *b)
{
	return a != b;
}

// Wait for a specific task when a pool of threads is responsible for progress.
int aml_queue_sched_wait_async(struct aml_sched_data *data,
                               struct aml_task *task)
{
	int err;
	struct aml_queue_sched *sched = (struct aml_queue_sched *)data;
	struct aml_queue *q = sched->done_q;
	pthread_mutex_t *lock = &(sched->doneq_lock);
	void **match;

loop:
	pthread_mutex_lock(lock);
	while (aml_queue_len(q) == 0) {
		pthread_mutex_unlock(lock);
		sched_yield();
		goto loop;
	}

	err = aml_queue_find(q, task, comp_tasks, &match);

	if (err == AML_SUCCESS) {
		assert(aml_queue_take(q, match) == AML_SUCCESS);
		pthread_mutex_unlock(lock);
		return AML_SUCCESS;
	} else if (err == -AML_EDOM) {
		pthread_mutex_unlock(lock);
		goto loop;
	} else {
		pthread_mutex_unlock(lock);
		return err;
	}
}

// Wait for a specific task when the calling thread is responsible for progress.
int aml_queue_sched_wait(struct aml_sched_data *data, struct aml_task *task)
{
	int err;
	void **match;
	struct aml_queue_sched *sched = (struct aml_queue_sched *)data;

loop:
	// Look into done_q for tasks and return success if found.
	pthread_mutex_lock(&(sched->doneq_lock));
	err = aml_queue_find(sched->done_q, task, comp_tasks, &match);
	if (err == AML_SUCCESS) {
		assert(aml_queue_take(sched->done_q, match) == AML_SUCCESS);
		pthread_mutex_unlock(&(sched->doneq_lock));
		return AML_SUCCESS;
	}
	pthread_mutex_unlock(&(sched->doneq_lock));
	if (err != -AML_EDOM)
		return err;

	// Look into work_q for the task to do.
	pthread_mutex_lock(&(sched->workq_lock));
	err = aml_queue_find(sched->work_q, task, comp_tasks, &match);
	if (err == AML_SUCCESS) {
		struct aml_task *t = *(struct aml_task **)match;
		assert(aml_queue_take(sched->work_q, match) == AML_SUCCESS);
		pthread_mutex_unlock(&(sched->workq_lock));
		t->fn(t->in, t->out);
		return AML_SUCCESS;
	}
	pthread_mutex_unlock(&(sched->workq_lock));
	if (err != -AML_EDOM)
		return err;

	goto loop;
}

// Wait for any task when their is a pool of threads is responsible for
// progress.
struct aml_task *aml_queue_sched_wait_any_async(struct aml_sched_data *data)
{
	struct aml_task *task = NULL;
	struct aml_queue_sched *sched = (struct aml_queue_sched *)data;
	int busy;
	struct aml_task **busy_cmp[sched->nt];
	memset(busy_cmp, 0, sizeof(busy_cmp));

	// Poll done_q until a task is here.
	while (1) {
		// Pull one task.
		pthread_mutex_lock(&(sched->doneq_lock));
		task = aml_queue_pop(sched->done_q);
		if (task != NULL) {
			pthread_mutex_unlock(&(sched->doneq_lock));
			return task;
		}

		// Lock the entire scheduler state to see if there is work
		// to wait for.
		pthread_mutex_lock(&(sched->workq_lock));
		pthread_mutex_lock(&(sched->in_progress_lock));

		busy = aml_queue_len(sched->work_q) > 0 ||
		       memcmp(busy_cmp, sched->in_progress, sizeof(busy_cmp));
		pthread_mutex_unlock(&(sched->in_progress_lock));
		pthread_mutex_unlock(&(sched->doneq_lock));
		pthread_mutex_unlock(&(sched->workq_lock));

		// If there is no more work, then return NULL
		if (!busy)
			return NULL;

		// There is work to wait for, yield and loop
		sched_yield();
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
	pthread_mutex_unlock(&(sched->doneq_lock));

	if (task != NULL)
		return task;

	pthread_mutex_lock(&(sched->workq_lock));
	task = aml_queue_pop(sched->work_q);
	if (task != NULL)
		task->fn(task->in, task->out);
	pthread_mutex_unlock(&(sched->workq_lock));
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
		sched = AML_INNER_MALLOC_EXTRA(num_threads, struct aml_task *,
		                               num_threads * sizeof(pthread_t),
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
	if (num_threads) {
		data->in_progress = AML_INNER_MALLOC_GET_ARRAY(
		        sched, struct aml_task *, struct aml_sched,
		        struct aml_queue_sched, struct aml_sched_ops);
		data->threads = AML_INNER_MALLOC_GET_EXTRA(
		        sched, num_threads, struct aml_task *, struct aml_sched,
		        struct aml_queue_sched, struct aml_sched_ops);
		memset(data->in_progress, 0,
		       num_threads * sizeof(struct aml_task *));
	} else
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
	if (pthread_mutex_init(&data->in_progress_lock, NULL) != 0) {
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
	pthread_mutex_destroy(&data->in_progress_lock);

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
