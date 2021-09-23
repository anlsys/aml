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

int comp_int(const void *a, const void *b)
{
	int x = *(int *)a;
	int y = *(int *)b;

	if (x < y)
		return -1;
	if (x == y)
		return 0;
	return 1;
}

void test_err_conditions()
{
	struct aml_vector *vector;

	void *ret;
	const int key = 5;
	const size_t len = 1;

	// Test create
	assert(aml_vector_create(NULL, sizeof(int)) == -AML_EINVAL);
	assert(aml_vector_create(&vector, 0) == -AML_EINVAL);
	assert(aml_vector_create(&vector, sizeof(int)) == AML_SUCCESS);

	// Test resize
	assert(aml_vector_resize(NULL, 0) == -AML_EINVAL);

	// Test get
	assert(aml_vector_get(NULL, 0, NULL) == -AML_EINVAL);
	assert(aml_vector_get(vector, len, NULL) == -AML_EINVAL);
	assert(aml_vector_get(vector, len, &ret) == -AML_EDOM);

	// Test find
	assert(aml_vector_find(vector, &key, comp_int, NULL) ==
	       -AML_FAILURE);
	assert(aml_vector_find(NULL, &key, comp_int, NULL) ==
	       -AML_EINVAL);
	assert(aml_vector_find(vector, NULL, comp_int, NULL) == -AML_EINVAL);

	// Test sort
	assert(aml_vector_sort(NULL, comp_int) == -AML_EINVAL);
	assert(aml_vector_sort(vector, NULL) == -AML_EINVAL);

	// Test bsearch
	assert(aml_vector_sort(vector, comp_int) == AML_SUCCESS);
	assert(aml_vector_bsearch(vector, &key, comp_int, NULL) ==
	       -AML_FAILURE);
	assert(aml_vector_bsearch(NULL, &key, comp_int, NULL) ==
	       -AML_EINVAL);
	assert(aml_vector_bsearch(vector, NULL, comp_int, NULL) == -AML_EINVAL);

	// Test take
	assert(aml_vector_take(NULL, 0, NULL) == -AML_EINVAL);
	assert(aml_vector_take(vector, len, NULL) == -AML_EINVAL);
	assert(aml_vector_take(vector, len, &ret) == -AML_EDOM);

	// Test push
	assert(aml_vector_push_back(NULL, (void *)&key) == -AML_EINVAL);
	assert(aml_vector_push_back(vector, NULL) == -AML_EINVAL);

	// Test pop
	assert(aml_vector_pop_back(NULL, NULL) == -AML_EINVAL);
	assert(aml_vector_pop_back(vector, NULL) == -AML_EINVAL);
	assert(aml_vector_pop_back(vector, &ret) == -AML_EDOM);

	aml_vector_destroy(&vector);
}

void test_functional()
{
	struct aml_vector *vector;
	int test[6] = {0, 1, 2, -1, -2, -3};
	void *out;
	int val;
	size_t pos;

	assert(aml_vector_create(&vector, sizeof(int)) == AML_SUCCESS);

	// [0, 1, 2]
	for (int i = 0; i < 3; i++)
		assert(aml_vector_push_back(vector, &(test[i])) == AML_SUCCESS);

	// Make sure vector contains the right elements.
	for (int i = 0; i < 3; i++) {
		assert(aml_vector_get(vector, i, &out) == AML_SUCCESS);
		assert(*(int *)out == test[i]);
	}

	// Make sure resizing does not alter vector
	assert(aml_vector_resize(vector, 32) == AML_SUCCESS);
	for (int i = 0; i < 3; i++) {
		assert(aml_vector_get(vector, i, &out) == AML_SUCCESS);
		assert(*(int *)out == test[i]);
	}
	assert(aml_vector_resize(vector, 3) == AML_SUCCESS);

	// [0, 1, 2, -1, -2, -3]
	for (int i = 3; i < 6; i++)
		assert(aml_vector_push_back(vector, (void *)&test[i]) ==
		       AML_SUCCESS);
	// Make sure vector contains the right elements.
	for (int i = 0; i < 6; i++) {
		assert(aml_vector_get(vector, i, &out) == AML_SUCCESS);
		assert(*(int *)out == test[i]);
	}

	// [0, 1, 2, -1, -2, -3]
	val = -3;
	assert(aml_vector_find(vector, (void *)&val, comp_int, &pos) ==
	       AML_SUCCESS);
	assert(pos == 5);

	// [-3, -2, -1, 0, 1, 2]
	assert(aml_vector_sort(vector, comp_int) == AML_SUCCESS);
	// val = -3
	assert(aml_vector_bsearch(vector, (void *)&val, comp_int, &pos) ==
	       AML_SUCCESS);
	assert(pos == 0);

	// [-3, -2, -1, 0, 1]
	assert(aml_vector_pop_back(vector, (void *)&val) == AML_SUCCESS);
	assert(val == 2);

	// [-3, -1, 0, 1]
	assert(aml_vector_take(vector, 1, (void *)&val) == AML_SUCCESS);
	assert(val == -2);
	assert(aml_vector_get(vector, 1, &out) == AML_SUCCESS);
	assert(*(int *)out == -1);
	assert(aml_vector_get(vector, 3, &out) == AML_SUCCESS);
	assert(*(int *)out == 1);

	aml_vector_destroy(&vector);
}

int main(void)
{
	test_functional();
	test_err_conditions();
	return 0;
}
