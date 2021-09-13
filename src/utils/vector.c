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

int aml_vector_create(struct aml_vector **vector,
                      const size_t element_size,
                      size_t initial_len)
{
	struct aml_vector *a;

	if (vector == NULL || element_size == 0)
		return -AML_EINVAL;
	if (initial_len == 0)
		initial_len = 256;

	a = malloc(sizeof(*a));
	if (a == NULL)
		return -AML_ENOMEM;

	// Alloc one big buffer aligned on element size.
	a->buf = calloc(initial_len, element_size);
	if (a->buf == NULL) {
		free(a);
		return -AML_ENOMEM;
	}

	a->element_size = element_size;
	a->len = 0;
	a->alloc_len = initial_len;
	*vector = a;
	return AML_SUCCESS;
}

void aml_vector_destroy(struct aml_vector **vector)
{
	if (vector == NULL)
		return;

	free((*vector)->buf);
	free(*vector);
	*vector = NULL;
}

int aml_vector_resize(struct aml_vector *vector, size_t newlen)
{
	if (vector == NULL)
		return -AML_EINVAL;

	const size_t element_size = vector->element_size;

	if (newlen == 0)
		newlen = (2 * vector->alloc_len);

	// Shrink or Extend
	if (newlen < vector->len || (newlen > vector->alloc_len)) {
		void *ptr = realloc(vector->buf, newlen * element_size);
		if (ptr == NULL)
			return -AML_ENOMEM;
		vector->buf = ptr;
		vector->alloc_len = newlen;
	}
	return AML_SUCCESS;
}

size_t aml_vector_size(const struct aml_vector *vector)
{
	return vector->len;
}

#define AML_VECTOR_GET(vector, index)                                          \
	(void *)((char *)(vector)->buf + (vector)->element_size * (index))

int aml_vector_get(struct aml_vector *vector, const size_t index, void **out)
{
	if (vector == NULL)
		return -AML_EINVAL;

	if (index >= vector->len)
		return -AML_EDOM;

	if (out != NULL)
		*out = AML_VECTOR_GET(vector, index);
	return AML_SUCCESS;
}

int aml_vector_find(const struct aml_vector *vector,
                    const void *key,
                    int (*comp)(const void *, const void *),
                    size_t *pos)
{
	if (vector == NULL || comp == NULL || key == NULL)
		return -AML_EINVAL;

	for (size_t i = 0; i < vector->len; i++) {
		if (!comp(AML_VECTOR_GET(vector, i), key)) {
			if (pos != NULL)
				*pos = i;
			return AML_SUCCESS;
		}
	}

	return -AML_FAILURE;
}

int aml_vector_sort(struct aml_vector *vector,
                    int (*comp)(const void *, const void *))
{
	if (vector == NULL || comp == NULL)
		return -AML_EINVAL;

	qsort(vector->buf, vector->len, vector->element_size, comp);
	return AML_SUCCESS;
}

int aml_vector_bsearch(const struct aml_vector *vector,
                       const void *key,
                       int (*comp)(const void *, const void *),
                       size_t *pos)
{
	if (vector == NULL || comp == NULL || key == NULL)
		return -AML_EINVAL;

	void *result = bsearch(key, vector->buf, vector->len,
	                       vector->element_size, comp);

	if (result == NULL)
		return -AML_FAILURE;

	if (pos != NULL)
		*pos = ((char *)result - (char *)vector->buf) /
		       vector->element_size;

	return AML_SUCCESS;
}

int aml_vector_push(struct aml_vector *vector, const void *element)
{
	if (vector == NULL || element == NULL)
		return -AML_EINVAL;

	int err = AML_SUCCESS;

	if ((vector->len + 1) > vector->alloc_len)
		err = aml_vector_resize(vector, 0);
	if (err != AML_SUCCESS)
		return err;

	memcpy(AML_VECTOR_GET(vector, vector->len), element,
	       vector->element_size);
	vector->len++;
	return AML_SUCCESS;
}

int aml_vector_pop(struct aml_vector *vector, void *out)
{
	if (vector == NULL)
		return -AML_EINVAL;

	if (vector->len == 0)
		return -AML_EDOM;

	if (out != NULL)
		memcpy(out, AML_VECTOR_GET(vector, vector->len - 1),
		       vector->element_size);

	vector->len--;
	return AML_SUCCESS;
}

int aml_vector_take(struct aml_vector *vector, const size_t position, void *out)
{
	if (vector == NULL)
		return -AML_EINVAL;
	if (position >= vector->len)
		return -AML_EDOM;

	if (out != NULL)
		memcpy(out, AML_VECTOR_GET(vector, position),
		       vector->element_size);

	vector->len--;
	memmove(AML_VECTOR_GET(vector, position),
	        AML_VECTOR_GET(vector, position + 1),
	        (vector->len - position) * vector->element_size);

	return AML_SUCCESS;
}
