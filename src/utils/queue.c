/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include "aml.h"

struct aml_queue *aml_queue_create(const size_t max)
{
	struct aml_queue *q;

	q = AML_INNER_MALLOC_ARRAY(max, void *, struct aml_queue);
	if (q == NULL)
		return NULL;
	q->max = max;
	q->head = 0;
	q->tail = 0;
	q->elems = AML_INNER_MALLOC_GET_ARRAY(q, void *, struct aml_queue);
	return q;
}

void aml_queue_clear(struct aml_queue *q)
{
	if (q != NULL) {
		q->head = 0;
		q->tail = 0;
	}
}

void aml_queue_destroy(struct aml_queue *q)
{
	free(q);
}

void **aml_queue_head(const struct aml_queue *q)
{
	if (q->max == 0 || aml_queue_len(q) == 0)
		return NULL;
	return &q->elems[q->head];
}

void **aml_queue_next(const struct aml_queue *q, const void **current)
{
	const void **elems = (const void **)q->elems;
	if (q == NULL || q->tail == q->head)
		return NULL;
	// Out of bounds return null
	if (current < elems || current >= elems + q->max)
		return NULL;
	// Empty Queue
	if (q->tail == q->head)
		return NULL;
	// End of queue
	if (current == elems + q->tail - 1)
		return NULL;
	// special value query head
	if (current == NULL)
		return aml_queue_head(q);
	// All element are contiguous.
	if (q->tail > q->head) {
		// Out of bounds
		if (current < elems + q->head || current >= elems + q->tail)
			return NULL;
		return (void **)current++;
	} else {
		if (current < elems + q->tail - 1)
			return (void **)current++;
		if (current < elems + q->head)
			return NULL;
		if (current < elems + q->max - 1)
			return (void **)current++;
		return (void **)elems;
	}
}

static struct aml_queue *aml_queue_extend(struct aml_queue *q)
{
	const size_t len = q->max;
	const size_t head = q->head;
	const size_t tail = q->tail;

	q = realloc(q, AML_SIZEOF_ALIGNED_ARRAY(2 * len, void *,
	                                        struct aml_queue));
	if (q == NULL)
		return NULL;
	q->elems = AML_INNER_MALLOC_GET_ARRAY(q, void *, struct aml_queue);
	q->max = len * 2;

	// If element are contiguous, no need for memmove.
	if (head < tail)
		return q;

	// head is right to tail and smaller than tail then move it at the end.
	if (len - head < tail) {
		q->head = q->max - len + head;
		memmove(q->elems + q->head, q->elems + head,
		        (len - head) * sizeof(void *));
	}
	// tail is left to head and smaller than head then move it after head.
	else {
		memmove(q->elems + len, q->elems, tail * sizeof(void *));
		q->tail = len + tail;
	}

	return q;
}

size_t aml_queue_len(const struct aml_queue *q)
{
	if (q->tail > q->head)
		return q->tail - q->head;
	if (q->head > q->tail)
		return q->max - q->head + q->tail;
	return 0;
}

int aml_queue_push(struct aml_queue **q, void *element)
{
	struct aml_queue *r;

	if (q == NULL || *q == NULL)
		return -AML_EINVAL;
	r = *q;

	const size_t len = aml_queue_len(r);

	if (len >= r->max - 1) {
		r = aml_queue_extend(r);
		if (r == NULL)
			return -AML_ENOMEM;
		*q = r;
	}

	r->elems[r->tail] = element;
	r->tail = (r->tail + 1) % r->max;

	return AML_SUCCESS;
}

void *aml_queue_pop(struct aml_queue *q)
{
	void *out;

	if (q == NULL || q->tail == q->head)
		return NULL;
	out = q->elems[q->head];
	q->head = (q->head + 1) % q->max;
	return out;
}

/**
 * Take an element out and stitch the circular buffer to
 * make elements contiguous again.
 **/
void *aml_queue_take(struct aml_queue *q, void *element)
{
	if (q == NULL || q->tail == q->head)
		return NULL;

	// queue is empty
	if (q->tail == q->head)
		return NULL;

	// All element are contiguous but the one removed.
	if (q->tail > q->head) {
		// move elements after the one removed by one to the left.
		for (size_t i = q->head; i < q->tail; i++) {
			if (q->elems[i] == element) {
				memmove(q->elems + i, q->elems + i + 1,
				        sizeof(void *) * (q->tail - i - 1));
				q->tail--;
				return element;
			}
		}
		return NULL;
	}

	// tail is before head
	if (q->tail < q->head) {
		// move elements after the one removed by one to the left,
		// when the element is between 0 and tail.
		for (size_t i = 0; i < q->tail; i++) {
			if (q->elems[i] == element) {
				memmove(q->elems + i, q->elems + i + 1,
				        sizeof(void *) * (q->tail - i - 1));
				q->tail--;
				return element;
			}
		}
		// move elements after the one removed by one to the left,
		// when the element is between head and end. Then move
		// element at index 0 to the end. Finally slide elements from
		// 1 to tail by one to the left.
		for (size_t i = q->head; i < q->max; i++) {
			if (q->elems[i] == element) {
				memmove(q->elems + i, q->elems + i + 1,
				        sizeof(void *) * (q->max - i - 1));
				q->elems[q->max - 1] = q->elems[0];
				if (q->tail > 0) {
					memmove(q->elems, q->elems + 1,
					        sizeof(void *) * (q->tail - 1));
					q->tail--;
				} else
					q->tail = q->max - 1;
				return element;
			}
		}
		return NULL;
	}
	return NULL;
}
