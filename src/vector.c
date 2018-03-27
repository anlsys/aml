#include <aml.h>
#include <assert.h>
#include <errno.h>

/*******************************************************************************
 * Vector type:
 * generic vector of elements, with contiguous allocation: elements are part of
 * the vector.
 * This type supports one unusual feature: elements must contain an int key, at
 * a static offset, and this key as configurable "null" value.
 ******************************************************************************/

int aml_vector_resize(struct aml_vector *vec, size_t newsize)
{
	assert(vec != NULL);
	/* we don't shrink */
	if(vec->nbelems > newsize)
		return 0;

	vec->ptr = realloc(vec->ptr, newsize * vec->sz);
	assert(vec->ptr != NULL);
	for(int i = vec->nbelems; i < newsize; i++)
	{
		int *k = AML_VECTOR_KEY_P(vec, i);
		*k = vec->na;
	}
	vec->nbelems = newsize;
	return 0;
}

size_t aml_vector_size(struct aml_vector *vec)
{
	assert(vec != NULL);
	return vec->nbelems;
}

/* returns pointer to elements at index id */
void *aml_vector_get(struct aml_vector *vec, int id)
{
	assert(vec != NULL);
	if(id != vec->na && id < vec->nbelems)
		return AML_VECTOR_ELT_P(vec, id);
	else
		return NULL;
}

/* return index of first element with key */
int aml_vector_find(struct aml_vector *vec, int key)
{
	assert(vec != NULL);
	for(int i = 0; i < vec->nbelems; i++)
	{
		int *k = AML_VECTOR_KEY_P(vec, i);
		if(*k == key)
			return i;
	}
	return vec->na;
}

void *aml_vector_add(struct aml_vector *vec)
{
	assert(vec != NULL);
	
	int idx = aml_vector_find(vec, vec->na);
	if(idx == vec->na)
	{
		/* exponential growth, good to amortize cost */
		idx = vec->nbelems;
		aml_vector_resize(vec, vec->nbelems *2);
	}
	return AML_VECTOR_ELT_P(vec, idx);
}

void aml_vector_remove(struct aml_vector *vec, void *elem)
{
	assert(vec != NULL);
	assert(elem != NULL);
	assert(elem >= vec->ptr && elem < AML_VECTOR_ELT_P(vec, vec->nbelems));

	int *k = AML_VECTOR_ELTKEY_P(vec, elem);
	*k = vec->na;
}

/*******************************************************************************
 * Init/destroy:
 ******************************************************************************/

int aml_vector_init(struct aml_vector *vec, size_t reserve, size_t size,
		    size_t key, int na)
{
	assert(vec != NULL);
	vec->sz = size;
	vec->off = key;
	vec->na = na;
	vec->nbelems = reserve;
	vec->ptr = calloc(reserve, size);
	assert(vec->ptr != NULL);
	for(int i = 0; i < vec->nbelems; i++)
	{
		int *k = AML_VECTOR_KEY_P(vec, i);
		*k = na;
	}
	return 0;
}

int aml_vector_destroy(struct aml_vector *vec)
{
	assert(vec != NULL);
	free(vec->ptr);
	return 0;
}
