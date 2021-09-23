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
	/** Number of elements in queue */
	size_t len;
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
int aml_queue_push(struct aml_queue *q, void *element);

/**
 * Get an element out of the queue.
 * @return NULL if queue is empty.
 **/
void *aml_queue_pop(struct aml_queue *q);

/**
 * Find a matching element in a queue.
 * @param[in] q: An initialized queue where to find matching element.
 * @param[in] key: The key to search for in the queue.
 * @param[in] comp: A comparison function returning 0 when elements dont
 * match.
 * @param[out] out: A pointer where to store the pointer to element in
 * queue.
 * @return -AML_EINVAL if q is NULL or comp is NULL.
 * @return -AML_EDOM if key was not found.
 * @return -AML_SUCCESS if key was found. If `out` is not NULL, then
 * the pointer to matching element is stored in `out`.
 */
int aml_queue_find(struct aml_queue *q,
                   const void *key,
                   int comp(const void *, const void *),
                   void ***out);

/**
 * Get element at some index (from head to tail) in the queue.
 * @param[in] q: An initialized queue.
 * @param[in] index: An index between 0 and queue length.
 * @param[out] out: A pointer where to store the pointer to element in
 * queue.
 * @return -AML_EINVAL if q is NULL.
 * @return -AML_EDOM if index is out of bounds.
 * @return -AML_SUCCESS otherwise. If `out` is not NULL, then the pointer
 * to the matching element is stored in `out`.
 */
int aml_queue_get(const struct aml_queue *q, size_t index, void ***out);

/**
 * Take an element out of the queue. The element to remove must
 * point to a valid spot in the queue. If not, `-AML_EDOM` is returned.
 * @param[in, out] q: An initialized queue containing `element`.
 * @param[in] element: The pointer of to the element to remove inside the
 * queue. The pointer must point somewhere in the queue.
 * @return -AML_EINVAL if q is NULL.
 * @return -AML_EDOM if element is not a valid pointer of the queue.
 * @return AML_SUCCESS if the element has been successfully removed.
 **/
int aml_queue_take(struct aml_queue *q, void **element);

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
