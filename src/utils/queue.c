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

	q = malloc(sizeof(*q));
	if (q == NULL)
		return NULL;

	q->elems = calloc(max, sizeof(*q->elems));
	if (q->elems == NULL) {
		free(q);
		return NULL;
	}

	q->max = max;
	q->head = 0;
	q->tail = 0;
	q->len = 0;

	return q;
}

void aml_queue_clear(struct aml_queue *q)
{
	if (q != NULL) {
		q->head = 0;
		q->tail = 0;
		q->len = 0;
	}
}

void aml_queue_destroy(struct aml_queue *q)
{
	free(q->elems);
	free(q);
}

void **aml_queue_head(const struct aml_queue *q)
{
	if (q->max == 0 || q->len == 0)
		return NULL;
	return &q->elems[q->head];
}

void **aml_queue_next(const struct aml_queue *q, const void **current)
{
	const void **elems = (const void **)q->elems;
	if (q == NULL || q->len == 0)
		return NULL;

	// Special value query head
	if (current == NULL)
		return q->elems[q->head];

	// Head is before tail.
	if (q->tail > q->head) {
		if (current >= (elems + q->head) &&
		    current < (elems + q->tail - 1))
			return (void **)current + 1;
	}
	// Tail is before head
	else if (q->tail < q->head) {
		// Between 0 and tail - 1.
		if (current >= elems && current < (elems + q->tail - 1))
			return (void **)current + 1;
		// Between head and max - 1
		if (current >= (elems + q->head) &&
		    current < (elems + q->max - 1))
			return (void **)current + 1;
		// At max - 1. Can have next if tail is not 0.
		if ((current + 1) == (elems + q->max) && q->tail > 0)
			return (void **)elems;
	}
	return NULL;
}

int aml_queue_extend(struct aml_queue *q)
{
	void **elems = realloc(q->elems, 2 * q->max * sizeof(*elems));
	if (elems == NULL)
		return -AML_ENOMEM;
	q->elems = elems;

	// If head is after tail, move tail after head.
	if (q->tail <= q->head && q->tail > 0)
		memmove(q->elems + q->max, q->elems, q->tail * sizeof(void *));

	q->tail = q->head + q->len;
	q->max *= 2;

	return AML_SUCCESS;
}

size_t aml_queue_len(const struct aml_queue *q)
{
	return q->len;
}

int aml_queue_push(struct aml_queue *q, void *element)
{
	if (q == NULL)
		return -AML_EINVAL;

	int err;

	if (q->len >= q->max) {
		err = aml_queue_extend(q);
		if (err != AML_SUCCESS)
			return err;
	}

	q->elems[q->tail] = element;
	q->tail = (q->tail + 1) % q->max;
	q->len++;

	return AML_SUCCESS;
}

void *aml_queue_pop(struct aml_queue *q)
{
	void *out;

	if (q == NULL || q->len == 0)
		return NULL;
	out = q->elems[q->head];
	q->head = (q->head + 1) % q->max;
	q->len--;
	return out;
}

int aml_queue_find(struct aml_queue *q,
                   const void *key,
                   int comp(const void *, const void *),
                   void ***out)
{
	size_t i;
	if (q == NULL || comp == NULL)
		return -AML_EINVAL;

	else if (q->len == 0)
		return -AML_EDOM;

	// Head is before tail.
	else if (q->tail > q->head) {
		for (i = q->head; i < q->tail; i++) {
			if (!comp(q->elems[i], key))
				goto success;
		}
	}

	// Tail is before head
	if (q->tail <= q->head) {
		for (i = 0; i < q->tail; i++) {
			if (!comp(q->elems[i], key))
				goto success;
		}
		for (i = q->head; i < q->max; i++) {
			if (!comp(q->elems[i], key))
				goto success;
		}
	}

	// Not found.
	return -AML_EDOM;

success:
	if (out != NULL)
		*out = &(q->elems[i]);
	return AML_SUCCESS;
}

int aml_queue_get(const struct aml_queue *q, size_t index, void ***out)
{
	if (q == NULL)
		return -AML_EINVAL;

	void **element;

	// Head is before tail.
	if (q->head < q->tail && (q->head + index) < q->tail) {
		element = q->elems + q->head + index;
		goto success;
	}

	// Tail is before head
	else if (q->tail <= q->head) {
		const size_t head_len = q->max - q->head;

		// Index is between head and max.
		if (index < head_len) {
			element = q->elems + q->head + index;
			goto success;
		}
		// Index is between 0 and tail.
		else if (index - head_len < q->tail) {
			element = q->elems + index - head_len;
			goto success;
		}
	}

	// index is out of bounds.
	return -AML_EDOM;

success:
	if (out != NULL)
		*out = element;
	return AML_SUCCESS;
}

int aml_queue_take(struct aml_queue *q, void **element)
{
	if (q == NULL)
		return -AML_EINVAL;

	// Element is between head and tail and head is before tail OR
	// Element is between 0 and tail and tail is before head.
	if (element < (q->elems + q->tail)) {
		if ((q->tail <= q->head && element >= q->elems) ||
		    (q->head < q->tail && element >= q->elems + q->head)) {
			const size_t n = (q->elems + q->tail) - element - 1;
			if (n > 0)
				memmove(element, element + 1,
				        n * sizeof(void *));
			q->tail--;
			q->len--;
			return AML_SUCCESS;
		}
	}

	// Element is between head and max and tail is before head.
	else if (q->tail <= q->head && element >= q->elems + q->head &&
	         element < q->elems + q->max) {
		const size_t n = element - (q->elems + q->head);
		if (n > 0)
			memmove(q->elems + q->head + 1, q->elems + q->head,
			        n * sizeof(void *));
		q->head = (q->head + 1) % q->max;
		q->len--;
		return AML_SUCCESS;
	}

	// element is not in the queue.
	return -AML_EDOM;
}
