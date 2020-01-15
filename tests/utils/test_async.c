/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
*******************************************************************************/

#define _POSIX_C_SOURCE 199309L // nanosleep

#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <unistd.h>
#include "aml.h"
#include "aml/utils/async.h"

//----------------------------------------------------------------------------//
// Task Mockup Implementation for Tests
//----------------------------------------------------------------------------//

struct aml_task_out *aml_task_mockup_work(struct aml_task_in *in)
{
	(void)in;
	struct timespec us;

	us.tv_sec = 0;
	us.tv_nsec = 1000 * (rand() % 10);
	nanosleep(&us, NULL);
	return NULL;
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

	struct aml_sched *as = aml_active_sched_create(4);

	assert(as != NULL);
	test_scheduler(as, 256);
	aml_active_sched_destroy(&as);

	as = aml_active_sched_create(0);
	assert(as != NULL);
	test_scheduler(as, 256);
	aml_active_sched_destroy(&as);
	return 0;
}
