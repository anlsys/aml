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
 * Generic array type storing contiguous elements in a flat buffer.
 * This array is optimized to align element on elements size boundary.
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
 * This is equivalent to read `array->len`.
 *
 * @param[in] array: an initialized array structure.
 * @return The number of elements in the array.
 * @return If array is NULL, -AML_EINVAL is returned.
 **/
ssize_t aml_array_size(const struct aml_array *array);

/**
 * Get a pointer to element at position `index` into the space provided in
 *`out`. `out` must have at list `array->element_size` space.
 *
 * @param[in] array: an initialized array structure.
 * @param[in] index: an integer between 0 and array size.
 * @return NULL if `array` is NULL or if `index` is out of bounds.
 * @return A pointer to the element in the array.
 **/
void *aml_array_get(struct aml_array *array, const size_t index);

/**
 * Find the first element matching key.
 *
 * @param[in] array: an initialized array structure.
 * @param[in] key: the key to look for.
 * @param[in] comp: A comparison function returning 0 when input elements match.
 * @return -AML_EINVAL if `array`, `key` or `comp` is NULL.
 * @return the index of the found element or -AML_FAILURE if not found.
 **/
ssize_t aml_array_find(const struct aml_array *array,
                       const void *key,
                       int (*comp)(const void *, const void *));

/**
 * Sort array elements.
 *
 * @param[in] array: an initialized array structure.
 * @param[in] comp: A comparison function like linux `qsort` comparison
 *function.
 * @return -AML_EINVAL if `array` or `comp` is NULL.
 * @return AML_SUCCESS.
 **/
int aml_array_sort(struct aml_array *array,
                   int (*comp)(const void *, const void *));

/**
 * Find the first element matching key in a sorted array.
 *
 * @param[in] array: an initialized array structure.
 * @param[in] key: the key to look for.
 * @param[in] comp: A comparison function like linux `qsort` comparison
 * function.
 * @return -AML_EINVAL if `array`, `key` or `comp` is NULL.
 * @return the index of the found element or -AML_FAILURE if not found.
 **/
ssize_t aml_array_bsearch(const struct aml_array *array,
                          const void *key,
                          int (*comp)(const void *, const void *));

/**
 * Resizes the array.
 *
 * @param[in, out] array: A pointer to an initialized array structure.
 * The pointer might be updated to point to a new allocation.
 * @param newsize: a new array length. If newsize is less than current length
 * the array is truncated to newsize. If newsize is more than current allocated
 * length, then the array is extended.
 * @return AML_SUCCESS if successful; -AML_ENOMEM otherwise.
 **/
int aml_array_resize(struct aml_array *array, size_t newsize);

/**
 * Append an element at the end of the array.
 * Array is automatically enlarged if needed.
 *
 * @param[in, out] array: A pointer to an initialized array structure.
 * The pointer might be updated to point to a new allocation if `array` needs
 * to be extended.
 * @param[in] element: The element to append to array. `element` must have a
 * size of at least `array->element_size`.
 * @return AML_SUCCESS on success, or -AML_ENOMEM if a reallocation failed.
 **/
int aml_array_push(struct aml_array *array, const void *element);

/**
 * Remove element at the end of the array.
 *
 * @param[in] array: A pointer to an initialized array structure.
 * @param[out] out: A memory area with size of at least `array->element_size` to
 * copy end element. If `out` is NULL, nothing is copied.
 * @return AML_SUCCESS on success, or -AML_EINVAL if a `array` is empty.
 **/
int aml_array_pop(struct aml_array *array, void *out);

/**
 * Remove and retrieve element at a specific position.
 * This method may require a large `memmove()` call.
 *
 * @param[in] array: A pointer to an initialized array structure.
 * @param[in] position: An index position in array.
 * @param[out] out: A memory area with size of at least `array->element_size` to
 * copy array element. If `out` is NULL, nothing is copied.
 * @return -AML_EINVAL if `array` is NULL.
 * @return -AML_EDOM if `position` is out of bounds.
 * @return AML_SUCCESS on success.
 **/
int aml_array_take(struct aml_array *array, const size_t position, void *out);

/**
 * Allocate and initialize an empty array.
 *
 * @param[out] array: A pointer to an uninitialized array structure.
 * @param[in] element_size: The size of elements in array.
 * @param[in] initial_size: The initial array allocation size in number of
 * elements. If `initial_size` is 0, `initial_size` is set to 256.
 * @return AML_SUCCESS on success.
 * @return -AML_ENOMEM if allocation failed.
 * @return -AML_EINVAL if `element_size` is 0 or `array` is NULL.
 **/
int aml_array_create(struct aml_array *array,
                     const size_t element_size,
                     size_t initial_size);

/**
 * Release memory occupied by an array. This is equivalent to a call to
 * `free()` on `array->buf`.
 *
 * @param[in, out] array: a array created by `aml_array_create()`.
 * `NULL` after return.
 **/
void aml_array_destroy(struct aml_array *array);

/**
 * @}
 **/

#ifdef __cplusplus
}
#endif

#endif // AML_ARRAY_H
