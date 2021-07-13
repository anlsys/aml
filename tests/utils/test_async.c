/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#define _POSIX_C_SOURCE 199309L // nanosleep

#include <assert.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include "aml.h"

//----------------------------------------------------------------------------//
// Task Mockup Implementation for Tests
//----------------------------------------------------------------------------//

void aml_task_mockup_work(struct aml_task_in *in, struct aml_task_out *out)
{
	(void)in;
	(void)out;
	long int us = 1000 * (rand() % 10);
	long int count = 0;
	for (long int i = 0; i < us; i++)
		count += i ^ (count + i * 42);
}

struct aml_task aml_task_mockup = {
        .in = NULL,
        .out = NULL,
        .data = NULL,
        .fn = aml_task_mockup_work,
};

//----------------------------------------------------------------------------//
// Tests
//----------------------------------------------------------------------------//

void test_scheduler(struct aml_sched *sched, const unsigned int nt)
{
	struct aml_task *t;
	struct aml_task tasks[nt];

	for (unsigned int i = 0; i < nt; i++)
		tasks[i] = aml_task_mockup;

	// Submit one task.
	assert(aml_sched_submit_task(sched, tasks) == AML_SUCCESS);
	t = aml_sched_wait_any(sched);
	assert(t == tasks);

	// Submit all tasks.x
	for (unsigned int i = 0; i < nt; i++)
		assert(aml_sched_submit_task(sched, tasks + i) == AML_SUCCESS);

	// Wait for one specific task.
	assert(aml_sched_wait_task(sched, tasks + (nt / 2)) == AML_SUCCESS);
	for (unsigned int i = 0; i < nt - 1; i++) {
		t = aml_sched_wait_any(sched);
		assert(t != NULL);
		assert(t >= tasks);
		assert(t < tasks + nt);
	}
}

//----------------------------------------------------------------------------//
// Main
//----------------------------------------------------------------------------//

int main(void)
{
	// set seed for tasks sleep.
	srand(0);

	struct aml_sched *as = aml_queue_sched_create(4);

	assert(as != NULL);
	test_scheduler(as, 256);
	aml_queue_sched_destroy(&as);

	as = aml_queue_sched_create(0);
	assert(as != NULL);
	test_scheduler(as, 256);
	aml_queue_sched_destroy(&as);
	return 0;
}
