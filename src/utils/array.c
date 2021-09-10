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

int aml_array_create(struct aml_array **array,
                     size_t element_size,
                     size_t initial_size)
{

	// Alloc one big buffer aligned on element size.
	const size_t n = 1 + sizeof(struct aml_array) / element_size;
	void *ptr = calloc(n + initial_size, element_size);
	if (ptr == NULL)
		return -AML_EINVAL;

	struct aml_array *a = (struct aml_array *)(ptr);

	a->buf = (void *)((char *)ptr + n * element_size);
	a->element_size = element_size;
	a->len = 0;
	a->size = initial_size;

	*array = a;
	return AML_SUCCESS;
}

void aml_array_destroy(struct aml_array **array)
{
	if (array != NULL && *array != NULL) {
		free(*array);
		*array = NULL;
	}
}

int aml_array_resize(struct aml_array **array, size_t newsize)
{
	struct aml_array *a = *array;
	const size_t element_size = a->element_size;
	const size_t n = 1 + sizeof(struct aml_array) / element_size;

	if (newsize == 0)
		newsize = (2 * a->size / element_size);

	// Shrink or Extend
	if (newsize < a->len || (newsize * element_size > a->size)) {
		void *ptr = reallocarray(*array, (n + newsize), element_size);
		if (ptr == NULL)
			return -AML_EINVAL;
		*array = (struct aml_array *)ptr;
	}
	return AML_SUCCESS;
}

ssize_t aml_array_size(const struct aml_array *array)
{
	if (array == NULL)
		return -AML_EINVAL;
	return array->len;
}

#define AML_ARRAY_GET(array, index)                                            \
	(void *)((char *)(array)->buf + (array)->element_size * (index))

int aml_array_get(struct aml_array *array, size_t index, void *out)
{
	if (array == NULL)
		return -AML_EINVAL;
	if (index >= array->len)
		return -AML_EDOM;
	memcpy(out, AML_ARRAY_GET(array, index), array->element_size);
	return AML_SUCCESS;
}

ssize_t aml_array_find(const struct aml_array *array,
                       void *key,
                       int (*comp)(const void *, const void *))
{
	if (array == NULL || comp == NULL || key == NULL)
		return -AML_EINVAL;
	for (size_t i = 0; i < array->len; i++)
		if (!comp(AML_ARRAY_GET(array, i), key))
			return i;
	return -AML_FAILURE;
}

int aml_array_sort(const struct aml_array *array,
                   int (*comp)(const void *, const void *))
{
	if (array == NULL || comp == NULL)
		return -AML_EINVAL;

	qsort(array->buf, array->len, array->element_size, comp);
	return AML_SUCCESS;
}

ssize_t aml_array_bsearch(const struct aml_array *array,
                          void *key,
                          int (*comp)(const void *, const void *))
{
	if (array == NULL || comp == NULL || key == NULL)
		return -AML_EINVAL;

	void *result =
	        bsearch(key, array->buf, array->len, array->element_size, comp);

	if (result == NULL)
		return -AML_FAILURE;

	return ((char *)result - (char *)array->buf) / array->element_size;
}

int aml_array_push(struct aml_array **array, void *element)
{
	if (array == NULL || element == NULL)
		return -AML_EINVAL;

	struct aml_array *a = *array;
	int err = AML_SUCCESS;

	if ((a->len + 1) * a->element_size > a->size)
		err = aml_array_resize(array, 0);
	if (err != AML_SUCCESS)
		return err;

	memcpy(AML_ARRAY_GET(a, a->len), element, a->element_size);
	a->len++;
	return AML_SUCCESS;
}

int aml_array_pop(struct aml_array *array, void *out)
{
	if (array == NULL)
		return -AML_EINVAL;
	if (out != NULL)
		memcpy(out, AML_ARRAY_GET(array, array->len - 1),
		       array->element_size);
	array->len--;
	return AML_SUCCESS;
}

int aml_array_take(struct aml_array *array, size_t position, void *out)
{
	if (array == NULL || position >= array->len)
		return -AML_EINVAL;

	if (out != NULL)
		memcpy(out, AML_ARRAY_GET(array, position),
		       array->element_size);

	memmove(AML_ARRAY_GET(array, position),
	        AML_ARRAY_GET(array, position + 1), array->len - position);

	return AML_SUCCESS;
}
