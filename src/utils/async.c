/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include <sched.h>
#include <pthread.h>
#include "aml.h"

//----------------------------------------------------------------------------//
// Generic API functions.
//----------------------------------------------------------------------------//

int aml_sched_submit_task(struct aml_sched *pool, struct aml_task *task)
{
	if (pool == NULL || pool->data == NULL ||
	    pool->ops == NULL || pool->ops->submit == NULL)
		return -AML_EINVAL;

	return pool->ops->submit(pool->data, task);
}

int aml_sched_wait_task(struct aml_sched *pool, struct aml_task *task)
{
	if (pool == NULL || pool->data == NULL ||
	    pool->ops == NULL || pool->ops->wait == NULL)
		return -AML_EINVAL;

	return pool->ops->wait(pool->data, task);
}

struct aml_task *aml_sched_wait_any(struct aml_sched *pool)
{
	if (pool == NULL || pool->data == NULL ||
	    pool->ops == NULL || pool->ops->wait_any == NULL)
		return NULL;

	return pool->ops->wait_any(pool->data);
}

//----------------------------------------------------------------------------//
// Simple task scheduler with pthread workers.
//----------------------------------------------------------------------------//

/**
 * @brief Scheduler with busy wait.
 *
 * aml_active_sched has a pool of threads never sleeping.
 * Threads will look in a work queue for new tasks and output done
 * work to a different queue.
 *
 * In order to avoid contention on work queue lock,
 * when a thread gets the lock, it is responsible for feeding other threads
 * with work.
 **/
struct aml_active_sched {
	/** task queue **/
	struct aml_queue *work_q;
	/** Active lock on work_q to keep threads awake **/
	pthread_mutex_t workq_lock;
	/** work done queue **/
	struct aml_queue *done_q;
	/** Passive lock on done_q **/
	pthread_mutex_t doneq_lock;
	/** Number of threads **/
	size_t nt;
	/** threads **/
	pthread_t *threads;
	/** For each thread a slot with the next task to run **/
	struct aml_task **ready;
	/** For each thread a slot with the task being processed **/
	struct aml_task **running;
	/** id incremented by threads to get uniq "ready" slots **/
	size_t tid;
};

int aml_active_sched_num_tasks(struct aml_sched *sched)
{
	if (sched == NULL || sched->data == NULL)
		return -AML_EINVAL;

	struct aml_active_sched *s;
	unsigned int n = 0;

	s = (struct aml_active_sched *)sched->data;
	// Lock queues to get the right count.
	pthread_mutex_lock(&s->workq_lock);
	pthread_mutex_lock(&s->doneq_lock);
	// Count queues content
	n += aml_queue_len(s->work_q);
	n += aml_queue_len(s->done_q);
	// Count threads slots content
	for (size_t i = 0; i < s->nt; i++) {
		n += s->ready[i] == NULL ? 0 : 1;
		n += s->running[i] == NULL ? 0 : 1;
	}
	// Unlock queues
	pthread_mutex_unlock(&s->workq_lock);
	pthread_mutex_unlock(&s->doneq_lock);
	return n;
}

static void *aml_active_sched_thread_fn(void *arg)
{
	int err;
	size_t ntasks; // Number of tasks in queue
	size_t nassigned;  // Number of tasks assigned to threads
	struct aml_active_sched *sched;
	struct aml_task **ready;
	struct aml_task **running;

	// assign uniq slots by locking on a queue
	sched = (struct aml_active_sched *)arg;
	pthread_mutex_lock(&sched->doneq_lock);
	ready = sched->ready + sched->tid;
	running = sched->running + sched->tid;
	sched->tid += 1;
	pthread_mutex_unlock(&sched->doneq_lock);

	// Forever until cancelled by a call to aml_active_sched_destroy()
loop:
	pthread_testcancel();
	nassigned = 0; // No task yet assigned.

	// If "ready" slot has a task, then run it and flush the work to done_q
	if (*ready != NULL) {
		*running = *ready; // swap with running state
		*ready = NULL;
		// Run task
		(*running)->out = (*running)->fn((*running)->in);
		// flush done work to done_q
		pthread_mutex_lock(&(sched->doneq_lock));
		aml_queue_push(&(sched->done_q), (void *)(*running));
		*running = NULL;
		pthread_mutex_unlock(&(sched->doneq_lock));
		// Go to loop start. Maybe another thread has fed us.
		goto loop;
	}

	// If there is no work to do, fetch some work for all threads
	else {
		// Try to lock work_q as long as "ready" slot is empty.
		while ((err = pthread_mutex_trylock(&(sched->workq_lock))) ==
		       EBUSY)
			// If ready slot is not empty anymore then run the task.
			if (ready != NULL)
				goto loop;
		if (err != 0) {
			perror("pthread_mutex_trylock");
			goto loop;
		}
		// We own the lock, let's populate empty threads slots.
		ntasks = aml_queue_len(sched->work_q);
		for (size_t i = 0; i < sched->nt && ntasks; i++)
			if (sched->ready[i] == NULL) {
				sched->ready[i] = aml_queue_pop(sched->work_q);
				ntasks--;
				nassigned++;
			}
		pthread_mutex_unlock(&(sched->workq_lock));
	}

	// If nothing has been assigned, not even to us, then it might be
	// good to yield to some other thread.
	if (nassigned == 0)
		sched_yield();
	// loop until cancelation.
	goto loop;
	return NULL;
}

// Submit task.
int aml_active_sched_submit(struct aml_sched_data *data, struct aml_task *task)
{
	struct aml_active_sched *sched = (struct aml_active_sched *)data;
	int err = AML_SUCCESS;

	pthread_mutex_lock(&(sched->workq_lock));
	err = aml_queue_push(&(sched->work_q), task);
	pthread_mutex_unlock(&(sched->workq_lock));
	return err;
}

// Wait for a specific task when a pool of threads is responsible for progress.
int aml_active_sched_wait_async(struct aml_sched_data *data,
				struct aml_task *task)
{
	int done = 0;
	struct aml_active_sched *sched = (struct aml_active_sched *)data;

	while (!done) {
		if (aml_queue_len(sched->done_q) == 0) {
			sched_yield();
			continue;
		}
		pthread_mutex_lock(&(sched->doneq_lock));
		done = aml_queue_take(sched->done_q, task) != NULL;
		pthread_mutex_unlock(&(sched->doneq_lock));
	}

	return AML_SUCCESS;
}

// Wait for a specific task when the calling thread is responsible for progress.
int aml_active_sched_wait(struct aml_sched_data *data, struct aml_task *task)
{
	struct aml_active_sched *sched = (struct aml_active_sched *)data;

	// Look into done_q for tasks and return success if found.
	pthread_mutex_lock(&(sched->doneq_lock));
	if (aml_queue_take(sched->done_q, task) != NULL) {
		pthread_mutex_unlock(&(sched->doneq_lock));
		return AML_SUCCESS;
	}

	// Look into work_q for the task to do, do it then return success.
	pthread_mutex_lock(&(sched->workq_lock));
	aml_queue_take(sched->work_q, task);
	task->out = task->fn(task->in);
	pthread_mutex_unlock(&(sched->workq_lock));
	pthread_mutex_unlock(&(sched->doneq_lock));
	return AML_SUCCESS;
}

// Wait for any task when their is a pool of threads is responsible for
// progress.
struct aml_task *aml_active_sched_wait_any_async(struct aml_sched_data *data)
{
	struct aml_task *task = NULL;
	struct aml_active_sched *sched = (struct aml_active_sched *)data;

	// Poll done_q until a task is here.
	while (task == NULL) {
		pthread_mutex_lock(&(sched->doneq_lock));
		if (aml_queue_len(sched->done_q) == 0) {
			pthread_mutex_unlock(&(sched->doneq_lock));
			sched_yield();
			continue;
		}
		task = aml_queue_pop(sched->done_q);
		pthread_mutex_unlock(&(sched->doneq_lock));
	}

	return task;
}

// Wait for any task when calling thread is responsible for progress.
struct aml_task *aml_active_sched_wait_any(struct aml_sched_data *data)
{
	struct aml_task *task = NULL;
	struct aml_active_sched *sched = (struct aml_active_sched *)data;

	pthread_mutex_lock(&(sched->doneq_lock));
	task = aml_queue_pop(sched->done_q);

	if (task != NULL)
		goto found_in_doneq;

	pthread_mutex_lock(&(sched->workq_lock));
	task = aml_queue_pop(sched->work_q);
	if (task != NULL)
		task->out = task->fn(task->in);

	pthread_mutex_unlock(&(sched->workq_lock));
found_in_doneq:
	pthread_mutex_unlock(&(sched->doneq_lock));
	return task;
}

struct aml_sched *aml_active_sched_create(const size_t num_threads)
{
	size_t i;
	struct aml_sched *sched;
	struct aml_active_sched *data;
	struct aml_sched_ops *ops;
	const size_t qlen = num_threads == 0 ? 64 : num_threads * 8;

	// Alloc struct + pthreads + tasks slots
	if (num_threads > 0)
		sched = AML_INNER_MALLOC_EXTRA(num_threads * 2,
					       struct aml_task*,
					       num_threads * sizeof(pthread_t),
					       struct aml_sched,
					       struct aml_active_sched,
					       struct aml_sched_ops);
	else
		sched = AML_INNER_MALLOC(struct aml_sched,
					 struct aml_active_sched,
					 struct aml_sched_ops);

	if (sched == NULL) {
		aml_errno = -AML_ENOMEM;
		return NULL;
	}

	// assign data field
	data = AML_INNER_MALLOC_GET_FIELD(sched, 2,
					  struct aml_sched,
					  struct aml_active_sched,
					  struct aml_sched_ops);
	sched->data = (struct aml_sched_data *) data;

	// assign ops field
	ops = AML_INNER_MALLOC_GET_FIELD(sched, 3,
					 struct aml_sched,
					 struct aml_active_sched,
					 struct aml_sched_ops);
	if (num_threads != 0) {
		ops->submit = aml_active_sched_submit;
		ops->wait = aml_active_sched_wait_async;
		ops->wait_any = aml_active_sched_wait_any_async;
	} else {
		ops->submit = aml_active_sched_submit;
		ops->wait = aml_active_sched_wait;
		ops->wait_any = aml_active_sched_wait_any;
	}
	sched->ops = ops;

	// Assign ready, running and threads field in data.
	if (num_threads) {
		data->ready =
			AML_INNER_MALLOC_GET_ARRAY(sched, struct aml_task*,
						   struct aml_sched,
						   struct aml_active_sched,
						   struct aml_sched_ops);
		data->running = data->ready + num_threads;
		data->threads =
			AML_INNER_MALLOC_GET_EXTRA(sched,
						   num_threads * 2,
						   struct aml_task*,
						   struct aml_sched,
						   struct aml_active_sched,
						   struct aml_sched_ops);

		// Initialize task slots.
		for (i = 0; i < num_threads; i++) {
			data->ready[i] = NULL;	// No work to do
			data->running[i] = NULL; // No work done
		}
	} else {
		data->threads = NULL;
		data->ready = NULL;
		data->running = NULL;
	}

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
	data->tid = 0;

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
				   aml_active_sched_thread_fn, data) != 0)
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

void aml_active_sched_destroy(struct aml_sched **sched)
{
	struct aml_active_sched *data;

	if (sched == NULL || *sched == NULL)
		return;

	data = (struct aml_active_sched *) (*sched)->data;
	for (size_t i = 0; i < data->nt; i++)
		pthread_cancel(data->threads[i]);
	for (size_t i = 0; i < data->nt; i++)
		pthread_join(data->threads[i], NULL);
	pthread_mutex_destroy(&data->workq_lock);
	pthread_mutex_destroy(&data->doneq_lock);
	aml_queue_destroy(data->work_q);
	aml_queue_destroy(data->done_q);

	free(*sched);
	*sched = NULL;
}
