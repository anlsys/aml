/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
*******************************************************************************/

#ifndef AML_VECTOR_H
#define AML_VECTOR_H

/**
 * @defgroup aml_vector "AML Vector API"
 * @brief AML Vector API
 *
 * Generic vector type:
 * Vector of nbelems, each of size sz, with a comparison key at offset off
 * @{
 **/

/** Pointer to the key within element "e" of a vector "v".  **/
#define AML_VECTOR_ELTKEY_P(v, e) ((int *)(((intptr_t) e) + v->off))
/** Pointer to the key within element index "i" of a vector "v".  **/
#define AML_VECTOR_KEY_P(v, i) ((int *)(((intptr_t) v->ptr) + i*v->sz + v->off))
/** Pointer to the element index "i" of a vector "v".  **/
#define AML_VECTOR_ELT_P(v, i) ((void *)(((intptr_t) v->ptr) + i*v->sz))

/** AML vector structure **/
struct aml_vector {
	/** Flag telling that no value is yet assigned **/
	int na;
	/** Number of elements stored in vector **/
	size_t nbelems;
	/** Size of elements in vector **/
	size_t sz;
	/**
	 * The offset within each element where the
	 * key (of type int) is stored.
	 **/
	size_t off;
	/** Pointer to elements in vector **/
	void *ptr;
};

/**
 * Provides the total number of elements in the vector, including currently
 * unused ones.
 * @param vector: an initialized vector structure.
 * @return the number of elements in the vector.
 **/
size_t aml_vector_size(const struct aml_vector *vector);
/**
 * Provides a pointer of element with index "index" within the vector.
 * @param vector: an initialized vector structure.
 * @param index: a valid index within "vector".  The index must not equal
 *        "na" and must be lower than the size of the vector.
 * @return a pointer to the requested element.
 **/
void *aml_vector_get(struct aml_vector *vector, int index);
/**
 * Find the first element with a particular key.
 * @param vector: an initialized vector structure.
 * @param key: the key to look for.
 * @return the index of the found element or "na" if not found.
 **/
int aml_vector_find(const struct aml_vector *vector, int key);
/**
 * Resizes the vector.  The keys of the newly allocated elements are set to the
 * @param na value.
 * @param vector: an initialized vector structure.
 * @param newsize: a new vector size.  Only sizes greater than the current one
 *        will be honored; smaller sizes will result in a no-op.
 * @return 0 if successful; an error code otherwise.
 **/
int aml_vector_resize(struct aml_vector *vector, size_t newsize);
/**
 * Provides the pointer to the first unused element.  If the vector is full,
 * it automatically gets enlarged.
 * @param vector: an initialized vector structure.
 * @return the pointer to the first unused element.
 **/
void *aml_vector_add(struct aml_vector *vector);
/**
 * Removes an element from the vector.  The key of the element is set to the
 * "na" value.
 * @param vector: an initialized vector structure.
 * @param elem: an element within the vector.
 **/
void aml_vector_remove(struct aml_vector *vector, void *elem);

/**
 * Allocate and Initialize a vector. Allocates elements and sets their keys to
 * the "na" value.
 *
 * @param vector: the address of a pointer to a struct aml_vector used as a
 * return value.
 * @param num: the number of elements to allocate.
 * @param size: the size of each individual element.
 * @param key: the offset within each element where the key (of type int)
 * is stored.
 * @param na: a "null" key value used to indicate an unused element.
 * @return 0 if successful; an error code otherwise.
 **/
int aml_vector_create(struct aml_vector **vector, size_t num, size_t size,
		      size_t key, int na);

/**
 * Finalize and free a struct aml_vector.
 *
 * @param vector: a vector created by aml_vector_create. NULL after return.
 **/
void aml_vector_destroy(struct aml_vector **vector);

/**
 * @}
 **/

#endif //AML_VECTOR_H

