#ifndef AML_VECTOR_H
#define AML_VECTOR_H

/*******************************************************************************
 * Generic vector type:
 * Vector of nbelems, each of size sz, with a comparison key at offset off
 ******************************************************************************/

/* Pointer to the key within element "e" of a vector "v".  */
#define AML_VECTOR_ELTKEY_P(v,e) ((int *)(((intptr_t) e) + v->off))
/* Pointer to the key within element index "i" of a vector "v".  */
#define AML_VECTOR_KEY_P(v,i) ((int *)(((intptr_t) v->ptr) + i*v->sz + v->off))
/* Pointer to the element index "i" of a vector "v".  */
#define AML_VECTOR_ELT_P(v,i) ((void *)(((intptr_t) v->ptr) + i*v->sz))

struct aml_vector {
	int na;
	size_t nbelems;
	size_t sz;
	size_t off;
	void *ptr;
};

/* not needed, here for consistency */
#define AML_VECTOR_DECL(name) struct vector ##name;
#define AML_VECTOR_ALLOCSIZE (sizeof(struct vector))

/*
 * Provides the total number of elements in the vector, including currently
 * unused ones.
 * "vector": an initialized vector structure.
 * Returns the number of elements in the vector.
 */
size_t aml_vector_size(const struct aml_vector *vector);
/*
 * Provides a pointer of element with index "index" within the vector.
 * "vector": an initialized vector structure.
 * "index": a valid index within "vector".  The index must not equal "na" and
 *          must be lower than the size of the vector.
 * Returns a pointer to the requested element.
 */
void *aml_vector_get(struct aml_vector *vector, int index);
/*
 * Find the first element with a particular key.
 * "vector": an initialized vector structure.
 * "key": the key to look for.
 * Returns the index of the found element or "na" if not found.
 */
int aml_vector_find(const struct aml_vector *vector, int key);
/*
 * Resizes the vector.  The keys of the newly allocated elements are set to the
 * "na" value.
 * "vector": an initialized vector structure.
 * "newsize": a new vector size.  Only sizes greater than the current one will
 *            be honored; smaller sizes will result in a no-op.
 * Returns 0 if successful; an error code otherwise.
 */
int aml_vector_resize(struct aml_vector *vector, size_t newsize);
/*
 * Provides the pointer to the first unused element.  If the vector is full,
 * it automatically gets enlarged.
 * "vector": an initialized vector structure.
 * Returns the pointer to the first unused element.
 */
void *aml_vector_add(struct aml_vector *vector);
/*
 * Removes an element from the vector.  The key of the element is set to the
 * "na" value.
 * "vector": an initialized vector structure.
 * "elem": an element within the vector.
 */
void aml_vector_remove(struct aml_vector *vector, void *elem);

/*
 * Initializes a vector.  Allocates elements and sets their keys to the "na"
 * value.
 * "vector": an allocated vector structure.
 * "num": the number of elements to allocate.
 * "size": the size of each individual element.
 * "key": the offset within each element where the key (of type int) is stored.
 * "na": a "null" key value used to indicate an unused element.
 * Returns 0 if successful; an error code otherwise.
 */
int aml_vector_init(struct aml_vector *vector, size_t num, size_t size,
		    size_t key, int na);
/*
 * Tears down an initialized vector.  Releases the memory buffer holding the
 * elements.
 * "vector": an initialized vector structure.
 * Returns 0 if successful; an error code otherwise.
 */
int aml_vector_destroy(struct aml_vector *vector);

#endif //AML_VECTOR_H

