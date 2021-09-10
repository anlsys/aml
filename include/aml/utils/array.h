/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#ifndef AML_ARRAY_H
#define AML_ARRAY_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup aml_array "AML Array API"
 * @brief AML Array API
 *
 * Generic array type:
 * Array of nbelems, each of size sz, with a comparison key at offset off.
 * Elements are not stored directly in the array, but are allocated one by one.
 * @{
 **/

/** AML array structure **/
struct aml_array {
	/** Array storage: buf_size **/
	void *buf;
	/** Number of elements in array **/
	size_t len;
	/** Allocated size for buf in bytes **/
	size_t size;
	/** Size of array elements **/
	size_t element_size;
};

/**
 * Provides the total number of elements in the array.
 * @param array: an initialized array structure.
 * @return the number of elements in the array.
 **/
ssize_t aml_array_size(const struct aml_array *array);

/**
 * Provides a pointer of element at index "index" within the array.
 * @param array: an initialized array structure.
 * @param index: an integer between 0 and array size - 1.
 * @return a pointer to the requested element.
 **/
int aml_array_get(struct aml_array *array, size_t index, void *out);

/**
 * Find the first element matching key.
 * @param array: an initialized array structure.
 * @param key: the key to look for.
 * @param comp: A comparison function returning 0 when input elements match.
 * @return the index of the found element or -1 if not found.
 **/
ssize_t aml_array_find(const struct aml_array *array,
                       void *key,
                       int (*comp)(const void *, const void *));

int aml_array_sort(const struct aml_array *array,
                   int (*comp)(const void *, const void *));

ssize_t aml_array_bsearch(const struct aml_array *array,
                          void *key,
                          int (*comp)(const void *, const void *));

/**
 * Resizes the array.
 * @param array: an initialized array structure.
 * @param newsize: a new array size.  Only sizes greater than the current one
 *        will be honored; smaller sizes will result in a no-op.
 * @return 0 if successful; an error code otherwise.
 **/
int aml_array_resize(struct aml_array **array, size_t newsize);

/**
 * Provides the pointer to the first unused element.  If the array is full,
 * it automatically gets enlarged.
 * @param array: an initialized array structure.
 * @return the pointer to the first unused element.
 **/
int aml_array_push(struct aml_array **array, void *element);

int aml_array_pop(struct aml_array *array, void *out);

int aml_array_take(struct aml_array *array, size_t position, void *out);

/**
 * Allocate and Initialize a array. Allocates elements and sets their keys to
 * the "na" value.
 *
 * @param array: the address of a pointer to a struct aml_array used as a
 * return value.
 * @param num: the number of elements to allocate.
 * @param size: the size of each individual element.
 * @param key: the offset within each element where the key (of type int)
 * is stored.
 * @param na: a "null" key value used to indicate an unused element.
 * @return 0 if successful; an error code otherwise.
 **/
int aml_array_create(struct aml_array **array,
                     size_t element_size,
                     size_t initial_size);

/**
 * Finalize and free a struct aml_array.
 *
 * @param array: a array created by aml_array_create. NULL after return.
 **/
void aml_array_destroy(struct aml_array **array);

/**
 * @}
 **/

#ifdef __cplusplus
}
#endif

#endif // AML_ARRAY_H
