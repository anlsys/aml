/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
*******************************************************************************/

#include "aml.h"
#include <assert.h>

#define SIZE 64
int mem[SIZE];
struct aml_queue *q;

int main(void)
{
	int tail = 0, head = 0;

	for (int i = 0; i < SIZE; i++)
		mem[i] = i;

	q = aml_queue_create(SIZE);
	assert(q != NULL);
	assert(aml_queue_len(q) == 0);
	aml_queue_clear(q);
	assert(aml_queue_len(q) == 0);

	// Check insertion
	for (int i = 0; i < SIZE * 2; i++, tail++) {
		void *e = &mem[tail % SIZE];

		aml_queue_push(&q, e);
		assert((int)aml_queue_len(q) == (tail - head) + 1);
		assert(q->elems[q->tail - 1] == e);
	}

	// Check that popped elements order
	for (int i = 0; i < SIZE; i++, head++) {
		void *e = aml_queue_pop(q);

		assert(e == &mem[head % SIZE]);
		assert((int)aml_queue_len(q) == (tail - head) - 1);
	}

	// reinsert to imply resize
	for (int i = 0; i < SIZE * 2; i++, tail++) {
		aml_queue_push(&q, (void *)(&mem[tail % SIZE]));
		assert((int)aml_queue_len(q) == (tail - head) + 1);
	}

	// Check take specific elements
	assert(aml_queue_take(q, NULL) == NULL);
	for (int i = 0; i < SIZE; i++, head++)
		assert(aml_queue_take(q, mem+i) == mem+i);

	// Check empty queue.
	int len = aml_queue_len(q);

	assert(len == SIZE*2);
	for (int i = 0; i < len; i++, head++) {
		void *e = aml_queue_pop(q);

		assert(e == &mem[head % SIZE]);
		assert((int)aml_queue_len(q) == (tail - head) - 1);
	}
	assert(aml_queue_pop(q) == NULL);

	free(q);
	return 0;
}
