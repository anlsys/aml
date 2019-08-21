/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
*******************************************************************************/

#include "aml.h"
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
	if (vec->nbelems > newsize)
		return 0;

	vec->ptr = realloc(vec->ptr, newsize * vec->sz);
	assert(vec->ptr != NULL);
	for (size_t i = vec->nbelems; i < newsize; i++) {
		vec->ptr[i] = calloc(1, vec->sz);
		assert(vec->ptr[i] != NULL);
		int *k = AML_VECTOR_KEY_P(vec, i);
		*k = vec->na;
	}
	vec->nbelems = newsize;
	return 0;
}

size_t aml_vector_size(const struct aml_vector *vec)
{
	assert(vec != NULL);
	return vec->nbelems;
}

/* returns pointer to elements at index id */
void *aml_vector_get(struct aml_vector *vec, int id)
{
	assert(vec != NULL);
	if (id == vec->na || id < 0)
		return NULL;

	size_t idx = (size_t)id;

	if (idx < vec->nbelems)
		return AML_VECTOR_ELT_P(vec, idx);
	else
		return NULL;
}

int aml_vector_getid(struct aml_vector *vec, void *elem)
{
	assert(vec != NULL);
	assert(elem != NULL);

	for (size_t i = 0; i < vec->nbelems; i++)
		if (vec->ptr[i] == elem)
			return i;
	return vec->na;
}

/* return index of first element with key */
int aml_vector_find(const struct aml_vector *vec, int key)
{
	assert(vec != NULL);
	for (size_t i = 0; i < vec->nbelems; i++) {
		int *k = AML_VECTOR_KEY_P(vec, i);

		if (*k == key)
			return i;
	}
	return vec->na;
}

void *aml_vector_add(struct aml_vector *vec)
{
	assert(vec != NULL);
	int idx = aml_vector_find(vec, vec->na);

	if (idx == vec->na) {
		/* exponential growth, good to amortize cost */
		idx = vec->nbelems;
		aml_vector_resize(vec, vec->nbelems * 2);
	}
	return AML_VECTOR_ELT_P(vec, idx);
}

void aml_vector_remove(struct aml_vector *vec, void *elem)
{
	assert(vec != NULL);
	assert(elem != NULL);

	int *k = AML_VECTOR_ELTKEY_P(vec, elem);
	*k = vec->na;
}

/*******************************************************************************
 * Create/Destroy:
 ******************************************************************************/

int aml_vector_create(struct aml_vector **vec, size_t reserve, size_t size,
		      size_t key, int na)
{
	struct aml_vector *ret = NULL;
	void **ptr;

	if (vec == NULL)
		return -AML_EINVAL;

	ret = malloc(sizeof(struct aml_vector));
	if (ret == NULL) {
		*vec = NULL;
		return -AML_ENOMEM;
	}

	ptr = calloc(reserve, sizeof(void *));
	if (ptr == NULL) {
		free(ret);
		*vec = NULL;
		return -AML_ENOMEM;
	}

	for (size_t i = 0; i < reserve; i++) {
		ptr[i] = calloc(1, size);
		if (ptr[i] == NULL) {
			/* avoid issues with size_t and negative values */
			for (size_t j = 0; j + 1 <= i; j++)
				free(ptr[i]);
			free(ret);
			*vec = NULL;
			return -AML_ENOMEM;
		}
	}

	ret->sz = size;
	ret->off = key;
	ret->na = na;
	ret->nbelems = reserve;
	ret->ptr = ptr;
	for (size_t i = 0; i < ret->nbelems; i++) {
		int *k = AML_VECTOR_KEY_P(ret, i);
		*k = na;
	}

	*vec = ret;
	return 0;
}

void aml_vector_destroy(struct aml_vector **vec)
{
	struct aml_vector *v;

	if (vec == NULL)
		return;

	v = *vec;
	if (v == NULL)
		return;

	for (size_t i = 0; i < v->nbelems; i++)
		free(v->ptr[i]);

	free(v->ptr);
	free(v);
	*vec = NULL;
}
