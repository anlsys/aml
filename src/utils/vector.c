/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

/* error checking logic */
#define utarray_oom()                                                          \
	do {                                                                   \
		aml_errno = AML_ENOMEM;                                        \
		goto utarray_error;                                            \
	} while (0)

#include "aml.h"

#include "internal/utarray.h"

struct aml_vector {
	UT_array *array;
	UT_icd icd;
};

int aml_vector_create(struct aml_vector **vector, const size_t element_size)
{
	struct aml_vector *a;

	if (vector == NULL || element_size == 0)
		return -AML_EINVAL;

	a = malloc(sizeof(*a));
	if (a == NULL)
		return -AML_ENOMEM;

	a->icd.sz = element_size;
	a->icd.init = NULL;
	a->icd.copy = NULL;
	a->icd.dtor = NULL;

	utarray_new(a->array, &a->icd);
	*vector = a;
	return AML_SUCCESS;
utarray_error:
	return -AML_ENOMEM;
}

void aml_vector_destroy(struct aml_vector **vector)
{
	if (vector == NULL)
		return;

	utarray_free((*vector)->array);
	free(*vector);
	*vector = NULL;
}

int aml_vector_resize(struct aml_vector *vector, size_t newlen)
{
	if (vector == NULL)
		return -AML_EINVAL;

	utarray_resize(vector->array, newlen);
	return AML_SUCCESS;
utarray_error:
	return -AML_ENOMEM;
}

int aml_vector_length(const struct aml_vector *vector, size_t *length)
{
	if (vector == NULL || length == NULL)
		return -AML_EINVAL;

	*length = utarray_len(vector->array);
	return AML_SUCCESS;
}

int aml_vector_get(const struct aml_vector *vector, size_t index, void **out)
{
	if (vector == NULL || out == NULL)
		return -AML_EINVAL;

	*out = utarray_eltptr(vector->array, index);
	if (*out == NULL)
		return -AML_EDOM;
	return AML_SUCCESS;
}

int aml_vector_find(const struct aml_vector *vector,
                    const void *key,
                    int (*comp)(const void *, const void *),
                    size_t *pos)
{
	if (vector == NULL || comp == NULL || key == NULL)
		return -AML_EINVAL;

	void *elt;
	for (elt = utarray_front(vector->array); elt != NULL;
	     elt = utarray_next(vector->array, elt)) {
		if (!comp(key, elt))
			break;
	}
	if (elt == NULL)
		return -AML_FAILURE;

	*pos = utarray_eltidx(vector->array, elt);
	return AML_SUCCESS;
}

int aml_vector_sort(struct aml_vector *vector,
                    int (*comp)(const void *, const void *))
{
	if (vector == NULL || comp == NULL)
		return -AML_EINVAL;

	utarray_sort(vector->array, comp);
	return AML_SUCCESS;
}

int aml_vector_bsearch(const struct aml_vector *vector,
                       const void *key,
                       int (*comp)(const void *, const void *),
                       size_t *pos)
{
	if (vector == NULL || comp == NULL || key == NULL)
		return -AML_EINVAL;

	void *elt = utarray_find(vector->array, key, comp);
	if (elt == NULL)
		return -AML_FAILURE;

	*pos = utarray_eltidx(vector->array, elt);
	return AML_SUCCESS;
}

int aml_vector_push_back(struct aml_vector *vector, const void *element)
{
	if (vector == NULL || element == NULL)
		return -AML_EINVAL;

	utarray_push_back(vector->array, element);
	return AML_SUCCESS;
utarray_error:
	return -AML_ENOMEM;
}

int aml_vector_pop_back(struct aml_vector *vector, void *out)
{
	if (vector == NULL || out == NULL)
		return -AML_EINVAL;

	void *back = utarray_back(vector->array);
	if (back == NULL)
		return -AML_EDOM;

	memcpy(out, back, vector->icd.sz);
	utarray_pop_back(vector->array);
	return AML_SUCCESS;
}

int aml_vector_take(struct aml_vector *vector, const size_t position, void *out)
{
	if (vector == NULL || out == NULL)
		return -AML_EINVAL;

	void *elt = utarray_eltptr(vector->array, position);
	if (elt == NULL)
		return -AML_EDOM;

	memcpy(out, elt, vector->icd.sz);

	utarray_erase(vector->array, position, 1);
	return AML_SUCCESS;
}
