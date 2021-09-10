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

int aml_array_create(struct aml_array *array,
                     const size_t element_size,
                     size_t initial_size)
{
	if (array == NULL || element_size == 0)
		return -AML_EINVAL;
	if (initial_size == 0)
		initial_size = 256;

	// Alloc one big buffer aligned on element size.
	array->buf = calloc(initial_size, element_size);
	if (array->buf == NULL)
		return -AML_ENOMEM;
	array->element_size = element_size;
	array->len = 0;
	array->size = initial_size * element_size;
	return AML_SUCCESS;
}

void aml_array_destroy(struct aml_array *array)
{
	if (array == NULL)
		return;
	free(array->buf);
	array->size = 0;
	array->len = 0;
}

int aml_array_resize(struct aml_array *array, size_t newsize)
{
	if (array == NULL)
		return -AML_EINVAL;

	const size_t element_size = array->element_size;

	if (newsize == 0)
		newsize = (2 * array->size / element_size);

	// Shrink or Extend
	if (newsize < array->len || (newsize * element_size > array->size)) {
		void *ptr = calloc(newsize, element_size);
		if (ptr == NULL)
			return -AML_ENOMEM;
		const size_t len = array->len < newsize ? array->len : newsize;
		memcpy(ptr, array->buf, len * element_size);
		free(array->buf);
		array->buf = ptr;
		array->size = element_size * newsize;
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

void *aml_array_get(struct aml_array *array, const size_t index)
{
	if (array == NULL)
		return NULL;

	if (index >= array->len)
		return NULL;
	return AML_ARRAY_GET(array, index);
}

ssize_t aml_array_find(const struct aml_array *array,
                       const void *key,
                       int (*comp)(const void *, const void *))
{
	if (comp == NULL || key == NULL)
		return -AML_EINVAL;
	for (size_t i = 0; i < array->len; i++)
		if (!comp(AML_ARRAY_GET(array, i), key))
			return i;
	return -AML_FAILURE;
}

int aml_array_sort(struct aml_array *array,
                   int (*comp)(const void *, const void *))
{
	if (comp == NULL)
		return -AML_EINVAL;

	qsort(array->buf, array->len, array->element_size, comp);
	return AML_SUCCESS;
}

ssize_t aml_array_bsearch(const struct aml_array *array,
                          const void *key,
                          int (*comp)(const void *, const void *))
{
	if (comp == NULL || key == NULL)
		return -AML_EINVAL;

	void *result =
	        bsearch(key, array->buf, array->len, array->element_size, comp);

	if (result == NULL)
		return -AML_FAILURE;

	return ((char *)result - (char *)array->buf) / array->element_size;
}

int aml_array_push(struct aml_array *array, const void *element)
{
	if (array == NULL || element == NULL)
		return -AML_EINVAL;

	int err = AML_SUCCESS;

	if ((array->len + 1) * array->element_size > array->size)
		err = aml_array_resize(array, 0);
	if (err != AML_SUCCESS)
		return err;

	memcpy(AML_ARRAY_GET(array, array->len), element, array->element_size);
	array->len++;
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

int aml_array_take(struct aml_array *array, const size_t position, void *out)
{
	if (array == NULL)
		return -AML_EINVAL;
	if (position >= array->len)
		return -AML_EDOM;

	if (out != NULL)
		memcpy(out, AML_ARRAY_GET(array, position),
		       array->element_size);

	memmove(AML_ARRAY_GET(array, position),
	        AML_ARRAY_GET(array, position + 1), array->len - position);
	array->len--;

	return AML_SUCCESS;
}
