/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include <assert.h>

#include "aml.h"

#define SIZE 64

int comp_ptr(const void *a, const void *b)
{
	return a != b;
}

void test_extend()
{
	struct aml_queue *q = aml_queue_create(2);

	// extend head before tail.
	aml_queue_push(q, (void *)0x0); // [ 0 ]
	aml_queue_push(q, (void *)0x1); // [ 0, 1]
	aml_queue_pop(q); // [  , 1]
	aml_queue_push(q, (void *)0x2); // [ 2, 1]
	aml_queue_push(q, (void *)0x3); // [ , 1, 2, 3]

	assert(q->max == 4);
	assert(q->head == 1);
	assert(q->tail == 0);
	assert(q->elems[1] == (void *)0x1);
	assert(q->elems[2] == (void *)0x2);
	assert(q->elems[3] == (void *)0x3);

	// extend tail before head.
	aml_queue_push(q, (void *)0x4); // [ 4, 1, 2, 3]
	aml_queue_pop(q); // [ 4,  , 2, 3]
	aml_queue_pop(q); // [ 4,  ,  , 3]
	aml_queue_push(q, (void *)0x5); // [ 4, 5,  , 3]
	aml_queue_push(q, (void *)0x6); // [ 4, 5, 6, 3]
	aml_queue_push(q, (void *)0x7); // [  ,  ,  , 3, 4, 5, 6, 7]

	assert(q->max == 8);
	assert(q->head == 3);
	assert(q->tail == 0);
	assert(q->elems[3] == (void *)0x3);
	assert(q->elems[4] == (void *)0x4);
	assert(q->elems[5] == (void *)0x5);
	assert(q->elems[6] == (void *)0x6);
	assert(q->elems[7] == (void *)0x7);

	aml_queue_destroy(q);
}

void test_take()
{
	struct aml_queue *q = aml_queue_create(2);

	// Error: Element is before head and head is before tail.
	aml_queue_push(q, (void *)0x0); // [ 0 ]
	aml_queue_push(q, (void *)0x1); // [ 0, 1]
	aml_queue_push(q, (void *)0x2); // [ 0, 1, 2, ]
	aml_queue_pop(q); // [  , 1, 2, ]
	assert(aml_queue_take(q, q->elems) == -AML_EDOM);

	// Error: Element is after tail and head is before tail.
	assert(aml_queue_take(q, q->elems + 3) == -AML_EDOM);

	// Element is between head and tail and head is before tail.
	// [  , 2,  ,  ]
	assert(aml_queue_take(q, q->elems + 1) == AML_SUCCESS);
	assert(q->len == 1);
	assert(q->head == 1);
	assert(q->tail == 2);

	// Error: Element is before 0 and tail is before head.
	aml_queue_push(q, (void *)0x3); // [  , 2, 3,  ]
	aml_queue_push(q, (void *)0x4); // [  , 2, 3, 4]
	aml_queue_pop(q); // [  ,  , 3, 4]
	aml_queue_push(q, (void *)0x5); // [ 5,  , 3, 4]
	assert(aml_queue_take(q, q->elems - 1) == -AML_EDOM);

	// Error: Element is between tail and head and tail is before head.
	assert(aml_queue_take(q, q->elems + 1) == -AML_EDOM);

	// Error: Element is after max and head and tail is before head.
	assert(aml_queue_take(q, q->elems + q->max) == -AML_EDOM);

	// Element is between 0 and tail and tail is before head.
	aml_queue_push(q, (void *)0x6); // [ 5, 6, 3, 4]
	// [ 6,  , 3, 4]
	assert(aml_queue_take(q, q->elems) == AML_SUCCESS);
	assert(q->len == 3);
	assert(q->head == 2);
	assert(q->tail == 1);
	assert(q->elems[0] == (void *)0x6);
	assert(q->elems[2] == (void *)0x3);
	assert(q->elems[3] == (void *)0x4);

	// Element is between head and max and tail is before head.
	// [ 6,  ,  , 3]
	assert(aml_queue_take(q, q->elems + 3) == AML_SUCCESS);
	assert(q->len == 2);
	assert(q->head == 3);
	assert(q->tail == 1);
	assert(q->elems[0] == (void *)0x6);
	assert(q->elems[3] == (void *)0x3);

	aml_queue_destroy(q);
}

void **assert_contains(struct aml_queue *q, void *key)
{
	void **match;
	assert(aml_queue_find(q, key, comp_ptr, &match) == AML_SUCCESS);
	assert(*match == key);
	return match;
}

void assert_insert(struct aml_queue *q, void *e)
{
	size_t len = aml_queue_len(q);
	assert(aml_queue_push(q, e) == AML_SUCCESS);
	assert(aml_queue_len(q) == len + 1);
	if (q->tail != 0)
		assert(q->elems[q->tail - 1] == e);
	else
		assert(q->elems[q->max - 1] == e);
	(void)assert_contains(q, e);
}

void assert_remove(struct aml_queue *q, size_t i)
{
	size_t len = aml_queue_len(q);
	void **e, *_e;
	assert(aml_queue_get(q, i, &e) == AML_SUCCESS);
	_e = *e;
	assert(aml_queue_take(q, e) == AML_SUCCESS);
	assert(aml_queue_len(q) == len - 1);
	assert(aml_queue_find(q, _e, comp_ptr, NULL) == -AML_EDOM);
}

void stress_test()
{
	const size_t num_tests = 4096;
	struct aml_queue *q = aml_queue_create(1);
	assert(q != NULL);

	assert_insert(q, (void *)0);
	for (size_t i = 1; i < num_tests; i++) {
		assert_insert(q, (void *)i);
		int rnd = rand() % (q->len * 2);
		if ((size_t)rnd < q->len)
			assert_remove(q, rnd);
	}

	aml_queue_destroy(q);
}

void test_queue()
{
	int mem[SIZE];
	struct aml_queue *q;
	int tail = 0, head = 0;

	for (int i = 0; i < SIZE; i++)
		mem[i] = i;

	q = aml_queue_create(SIZE);
	assert(q != NULL);
	assert(aml_queue_len(q) == 0);

	// Check insertion
	for (int i = 0; i < SIZE; i++, tail++)
		assert_insert(q, &mem[i]);

	// Check get elements match.
	for (int i = 0; i < SIZE; i++) {
		void **match;

		assert(aml_queue_get(q, i, &match) == AML_SUCCESS);
		assert(*match == &mem[i]);
	}

	// Check that popped elements order
	for (int i = 0; i < SIZE; i++, head++) {
		void *e = aml_queue_pop(q);

		assert(e == &mem[head % SIZE]);
		assert((int)aml_queue_len(q) == (tail - head) - 1);
	}

	// reinsert to force resize
	for (int i = 0; i < SIZE * 2; i++, tail++) {
		aml_queue_push(q, (void *)(&mem[tail % SIZE]));
		assert((int)aml_queue_len(q) == (tail - head) + 1);
	}

	// Check get elements match.
	for (int i = 0; i < SIZE * 2; i++) {
		void **match;

		assert(aml_queue_get(q, i, &match) == AML_SUCCESS);
		assert(*match == &mem[i % SIZE]);
	}

	// Check take specific elements
	assert(aml_queue_take(NULL, NULL) == -AML_EINVAL);
	assert(aml_queue_take(q, NULL) == -AML_EDOM);

	for (int i = 0; i < SIZE; i++, head++) {
		void **match = assert_contains(q, mem + i);
		assert(aml_queue_take(q, match) == AML_SUCCESS);
	}

	// Check empty queue.
	int len = aml_queue_len(q);

	assert(len == SIZE);
	for (int i = 0; i < len; i++, head++) {
		void *e = aml_queue_pop(q);

		assert(e == &mem[head % SIZE]);
		assert((int)aml_queue_len(q) == (tail - head) - 1);
	}
	assert(aml_queue_pop(q) == NULL);

	aml_queue_destroy(q);
}

int main(void)
{
	test_extend();
	test_take();
	test_queue();
	stress_test();
	return 0;
}
