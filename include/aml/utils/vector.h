/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#ifndef AML_VECTOR_H
#define AML_VECTOR_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup aml_vector "AML Vector API"
 * @brief AML Vector API
 *
 * Generic vector type storing contiguous elements in a flat buffer.
 * This vector is optimized to align element on elements size boundary.
 * Elements stored in the vector are supposed to be trivially copyable,
 * that is, they are not deep copied.
 *
 * @{
 **/

/** AML vector structure: some utils are kept opaque on purpose to avoid
 * publicly exporting underlying support code. **/
struct aml_vector;

/**
 * Provides the number of elements in the vector.
 *
 * @param[in] vector: an initialized vector structure.
 * @param[out] length: pointer to returned length of vector
 * @return AML_SUCCESS on success.
 * @return -AML_EINVAL if `vector` or `length` are NULL.
 **/
int aml_vector_length(const struct aml_vector *vector, size_t *length);

/**
 * Get a pointer to element at position `index`.
 *
 * @param[in] vector: an initialized vector structure.
 * @param[in] index: an integer in [[ 0, vector->len [[.
 * @param[in] out: A pointer where to store the pointer to vector element.
 * If `out` is NULL, nothing is stored.
 * @return AML_SUCCESS on success.
 * @return -AML_EINVAL if `vector` or `out` are NULL.
 * @return -AML_EDOM if `index` is out of bounds..
 **/
int aml_vector_get(const struct aml_vector *vector, size_t index, void **out);

/**
 * Find the first element matching key.
 *
 * @param[in] vector: an initialized vector structure.
 * @param[in] key: the key to look for.
 * @param[in] comp: A comparison function returning 0 when input elements match.
 * @param[out] pos: A pointer where to store the position of the found
 * element. If pos is NULL, nothing is stored.
 * @return -AML_EINVAL if `vector`, `key`, `pos`, or `comp` are NULL.
 * @return -AML_FAILURE if key was not found.
 * @return AML_SUCCESS if key was found.
 **/
int aml_vector_find(const struct aml_vector *vector,
                    const void *key,
                    int (*comp)(const void *, const void *),
                    size_t *pos);

/**
 * Sort vector elements.
 *
 * @param[in] vector: an initialized vector structure.
 * @param[in] comp: A comparison function like linux `qsort` comparison
 *function.
 * @return -AML_EINVAL if `vector` or `comp` is NULL.
 * @return AML_SUCCESS.
 **/
int aml_vector_sort(struct aml_vector *vector,
                    int (*comp)(const void *, const void *));

/**
 * Find the first element matching key in a sorted vector.
 *
 * @param[in] vector: an initialized vector structure.
 * @param[in] key: the key to look for.
 * @param[in] comp: A comparison function like linux `qsort` comparison
 * function.
 * @param[out] pos: A pointer where to store the position of the found
 * element. If pos is NULL, nothing is stored.
 * @return -AML_EINVAL if `vector`, `key`, `pos`, or `comp` are NULL.
 * @return -AML_FAILURE if key was not found.
 * @return AML_SUCCESS if key was found.
 **/
int aml_vector_bsearch(const struct aml_vector *vector,
                       const void *key,
                       int (*comp)(const void *, const void *),
                       size_t *pos);

/**
 * Resizes the vector.
 *
 * @param[in, out] vector: A pointer to an initialized vector structure.
 * The pointer might be updated to point to a new allocation.
 * @param newlen: a new vector length. If newlen is less than current length
 * the vector is truncated to newlen. If newlen is more than current
 * allocated length, then the vector is extended. 
 * @return AML_SUCCESS if successful; -AML_ENOMEM otherwise.
 **/
int aml_vector_resize(struct aml_vector *vector, size_t newlen);

/**
 * Append an element at the end of the vector.
 * Vector is automatically enlarged if needed.
 *
 * @param[in, out] vector: A pointer to an initialized vector structure.
 * The pointer might be updated to point to a new allocation if `vector`
 * needs to be extended.
 * @param[in] element: The element to append to vector.
 * @return AML_SUCCESS on success, or -AML_ENOMEM if a reallocation failed.
 **/
int aml_vector_push_back(struct aml_vector *vector, const void *element);

/**
 * Remove element at the end of the vector.
 *
 * @param[in] vector: A pointer to an initialized vector structure.
 * @param[out] out: A memory area with size of at
 * least `vector->element_size` to copy end element.
 * If `out` is NULL or `vector` is empty, nothing is copied.
 * @return AML_SUCCESS on success
 * @return -AML_EDOM if `vector` is empty.
 * @return -AML_EINVAL if `vector` is NULL.
 **/
int aml_vector_pop_back(struct aml_vector *vector, void *out);

/**
 * Remove and retrieve element at a specific position.
 * This method may require a large `memmove()` call.
 *
 * @param[in] vector: A pointer to an initialized vector structure.
 * @param[in] position: An index position in vector.
 * @param[out] out: A memory area with size of at least `vector->element_size`
 *to copy vector element. If `out` is NULL, nothing is copied.
 * @return -AML_EINVAL if `vector` or `out` are NULL.
 * @return -AML_EDOM if `position` is out of bounds.
 * @return AML_SUCCESS on success.
 **/
int aml_vector_take(struct aml_vector *vector,
                    const size_t position,
                    void *out);

/**
 * Allocate and initialize an empty vector.
 *
 * @param[out] vector: A pointer to an uninitialized vector structure.
 * @param[in] element_size: The size of elements in vector.
 * @return AML_SUCCESS on success.
 * @return -AML_ENOMEM if allocation failed.
 * @return -AML_EINVAL if `element_size` is 0 or `vector` is NULL.
 **/
int aml_vector_create(struct aml_vector **vector,
                      const size_t element_size);

/**
 * Release memory occupied by an vector.
 *
 * @param[in, out] vector: a vector created by `aml_vector_create()`.
 * `NULL` after return.
 **/
void aml_vector_destroy(struct aml_vector **vector);

/**
 * @}
 **/

#ifdef __cplusplus
}
#endif

#endif // AML_VECTOR_H
