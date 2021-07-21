/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#ifndef AML_QUEUE_H
#define AML_QUEUE_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup aml_queue "AML Queue API"
 * @brief AML Queue API
 *
 * Generic queue type allocated on heap:
 * Serial queue for pushing and poping pointers.
 * @{
 **/

/** queue struct definition **/
struct aml_queue {
	/** Maximum capacity. Is extended if reached **/
	size_t max;
	/** Index of head **/
	size_t head;
	/** Index of tail **/
	size_t tail;
	/** Elements in the queue **/
	void **elems;
};

/**
 * Create a queue with max pre-allocated space for max elements.
 * @param[in] max: The number of elements fitting in the queue before
 * trigerring a resize.
 * @return NULL if memory allocation failed.
 **/
struct aml_queue *aml_queue_create(const size_t max);

/**
 * Forget about elements stored in the queue.
 **/
void aml_queue_clear(struct aml_queue *q);

/**
 * Free queue. Calling free() directly on queue is ok.
 **/
void aml_queue_destroy(struct aml_queue *q);

/**
 * Get the number of elements in the queue.
 *@return 0 if q is NULL.
 **/
size_t aml_queue_len(const struct aml_queue *q);

/**
 * Add an element at the queue tail.
 * @return -AML_ENOMEM if queue needed to be extended and allocation failed.
 **/
int aml_queue_push(struct aml_queue **q, void *element);

/**
 * Get an element out of the queue.
 * @return NULL if queue is empty.
 **/
void *aml_queue_pop(struct aml_queue *q);

/**
 * Take an element out of the queue.
 * @return NULL if queue does not contain the element.
 **/
void *aml_queue_take(struct aml_queue *q, void *element);

/**
 * Get a pointer to first pointer element in the queue.
 *
 * @return NULL if the `q` is NULL or empty.
 * @return A pointer somewhere in `q->elems` pointing to an element
 * in the queue.
 **/
void **aml_queue_head(const struct aml_queue *q);

/**
 * Get next element after `current` in the queue.
 *
 * @param[in] q: The queue where `current` comes from.
 * @param[in] current: A pointer somewhere in `q->elems` or NULL.
 * If `current` is NULL, then the queue head is returned.
 * @return NULL if the `q` is NULL, if `q` is empty, if `current` is
 * the last pointer element in `q` or if `current` is not a pointer
 * somewhere in `q->elems` pointing to an element in the
 * queue.
 * @return A pointer somewhere in `q->elems` pointing to the element
 * following `current`.
 **/
void **aml_queue_next(const struct aml_queue *q, const void **current);
/**
 * @}
 **/

#ifdef __cplusplus
}
#endif

#endif // AML_QUEUE_H
