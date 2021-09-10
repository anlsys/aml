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

int main(void)
{
	struct aml_array array;
	int test[6] = {0, 1, 2, -1, -2, -3};
	int out;

	assert(aml_array_create(&array, sizeof(int), 2) == AML_SUCCESS);

	// [0, 1, 2]
	for (int i = 0; i < 3; i++)
		assert(aml_array_push(&array, &(test[i])) == AML_SUCCESS);

	// Make sure array contains the right elements.
	for (int i = 0; i < 3; i++)
		assert(*(int *)aml_array_get(&array, i) == test[i]);

	// Make sure resizing does not alter array
	assert(aml_array_resize(&array, 32) == AML_SUCCESS);
	for (int i = 0; i < 3; i++)
		assert(*(int *)aml_array_get(&array, i) == test[i]);

	// [0, 1, 2, -1, -2, -3]
	for (int i = 3; i < 6; i++)
		assert(aml_array_push(&array, (void *)&test[i]) == AML_SUCCESS);
	// Make sure array contains the right elements.
	for (int i = 0; i < 6; i++)
		assert(*(int *)aml_array_get(&array, i) == test[i]);

	// [0, 1, 2, -1, -2, -3]
	out = -3;
	assert(aml_array_find(&array, (void *)&out, comp_int) == 5);

	// [-3, -2, -1, 0, 1, 2]
	assert(aml_array_sort(&array, comp_int) == AML_SUCCESS);
	// out = -3
	assert(aml_array_bsearch(&array, (void *)&out, comp_int) == 0);

	// [-3, -2, -1, 0, 1]
	assert(aml_array_pop(&array, (void *)&out) == AML_SUCCESS);
	assert(out == 2);

	// [-3, -1, 0, 1, 2]
	assert(aml_array_take(&array, 1, (void *)&out) == AML_SUCCESS);
	assert(out == -2);
	assert(*(int *)aml_array_get(&array, 1) == -1);

	aml_array_destroy(&array);

	return 0;
}
