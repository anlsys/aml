/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
*******************************************************************************/

#include <assert.h>

#include "test_layout.h"
#include "aml/layout/dense.h"
#include "aml/layout/reshape.h"
#include "aml/layout/native.h"

static void test_reshape_discontiguous(void)
{
	int memory[7][6][5];

	size_t dims_col[3] = { 4, 5, 6 };
	size_t dims_row[3] = { 6, 5, 4 };

	size_t stride[3] = { 1, 1, 1 };

	size_t pitch_col[3] = { 5, 6, 7 };
	size_t pitch_row[3] = { 7, 6, 5 };

	size_t new_dims_col[5] = { 2, 2, 5, 2, 3 };
	size_t new_dims_row[5] = { 3, 2, 5, 2, 2 };

	size_t coords[5];
	void *ptr;

	int i = 0;

	for (int j = 0; j < 6; j++)
		for (int k = 0; k < 5; k++)
			for (int l = 0; l < 4; l++, i++)
				memory[j][k][l] = i;

	struct aml_layout *a, *b, *c;

	assert(aml_layout_dense_create(&a,
				       (void *)memory,
				       AML_LAYOUT_ORDER_COLUMN_MAJOR,
				       sizeof(int),
				       3, dims_col, stride,
				       pitch_col) == AML_SUCCESS);

	assert(aml_layout_reshape(a, &b, 5, new_dims_col) == AML_SUCCESS);

	aml_layout_reshape_create(&c,
				  a,
				  AML_LAYOUT_ORDER_COLUMN_MAJOR,
				  5, new_dims_col);
	aml_layout_fprintf(stderr, "test-reshape", c);
	i = 0;
	for (size_t j = 0; j < 3; j++)
		for (size_t k = 0; k < 2; k++)
			for (size_t l = 0; l < 5; l++)
				for (size_t m = 0; m < 2; m++)
					for (size_t n = 0; n < 2; n++, i++) {
						coords[0] = n;
						coords[1] = m;
						coords[2] = l;
						coords[3] = k;
						coords[4] = j;
						ptr = aml_layout_deref_safe(b,
						coords);
						assert(i == *(int *)ptr);
						ptr = aml_layout_deref_safe(c,
						coords);
						assert(i == *(int *)ptr);
					}

	free(a);
	free(b);
	free(c);

	assert(aml_layout_dense_create(&a,
				       (void *)memory,
				       AML_LAYOUT_ORDER_ROW_MAJOR,
				       sizeof(int),
				       3, dims_row, stride,
				       pitch_row) == AML_SUCCESS);

	assert(aml_layout_reshape(a, &b, 5, new_dims_row) == AML_SUCCESS);

	aml_layout_reshape_create(&c,
				  a,
				  AML_LAYOUT_ORDER_ROW_MAJOR, 5, new_dims_row);
	aml_layout_fprintf(stderr, "test-reshape", c);
	i = 0;
	for (size_t j = 0; j < 3; j++)
		for (size_t k = 0; k < 2; k++)
			for (size_t l = 0; l < 5; l++)
				for (size_t m = 0; m < 2; m++)
					for (size_t n = 0; n < 2; n++, i++) {
						coords[0] = j;
						coords[1] = k;
						coords[2] = l;
						coords[3] = m;
						coords[4] = n;
						ptr = aml_layout_deref_safe(b,
						coords);
						assert(i == *(int *)ptr);
						ptr = aml_layout_deref_safe(c,
						coords);
						assert(i == *(int *)ptr);
					}

	free(a);
	free(b);
	free(c);
}

static void test_reshape_strided(void)
{
	int memory[12][5][8];

	size_t dims_col[3] = { 4, 5, 6 };
	size_t dims_row[3] = { 6, 5, 4 };

	size_t stride[3] = { 2, 1, 2 };

	size_t pitch_col[3] = { 8, 5, 12 };
	size_t pitch_row[3] = { 12, 5, 8 };

	size_t new_dims_col[4] = { 2, 10, 2, 3 };
	size_t new_dims_row[4] = { 3, 2, 10, 2 };

	size_t coords[4];
	void *ptr;

	int i = 0;

	for (int j = 0; j < 6; j++)
		for (int k = 0; k < 5; k++)
			for (int l = 0; l < 4; l++, i++)
				memory[2 * j][1 * k][2 * l] = i;

	struct aml_layout *a, *b, *c;

	assert(aml_layout_dense_create(&a,
				       (void *)memory,
				       AML_LAYOUT_ORDER_COLUMN_MAJOR,
				       sizeof(int),
				       3, dims_col, stride,
				       pitch_col) == AML_SUCCESS);

	assert(aml_layout_reshape(a, &b, 4, new_dims_col) == AML_SUCCESS);

	aml_layout_reshape_create(&c,
				  a,
				  AML_LAYOUT_ORDER_COLUMN_MAJOR,
				  4, new_dims_col);

	i = 0;
	for (size_t j = 0; j < 3; j++)
		for (size_t k = 0; k < 2; k++)
			for (size_t l = 0; l < 10; l++)
				for (size_t m = 0; m < 2; m++, i++) {
					coords[0] = m;
					coords[1] = l;
					coords[2] = k;
					coords[3] = j;
					ptr = aml_layout_deref_safe(b, coords);
					assert(i == *(int *)ptr);
					ptr = aml_layout_deref_safe(c, coords);
					assert(i == *(int *)ptr);
				}

	free(a);
	free(b);
	free(c);

	assert(aml_layout_dense_create(&a,
				       (void *)memory,
				       AML_LAYOUT_ORDER_ROW_MAJOR,
				       sizeof(int),
				       3, dims_row, stride,
				       pitch_row) == AML_SUCCESS);

	assert(aml_layout_reshape(a, &b, 4, new_dims_row) == AML_SUCCESS);

	aml_layout_reshape_create(&c,
				  a,
				  AML_LAYOUT_ORDER_ROW_MAJOR, 4, new_dims_row);

	i = 0;
	for (size_t j = 0; j < 3; j++)
		for (size_t k = 0; k < 2; k++)
			for (size_t l = 0; l < 10; l++)
				for (size_t m = 0; m < 2; m++, i++) {
					coords[0] = j;
					coords[1] = k;
					coords[2] = l;
					coords[3] = m;
					ptr = aml_layout_deref_safe(b, coords);
					assert(i == *(int *)ptr);
					ptr = aml_layout_deref_safe(b, coords);
					assert(i == *(int *)ptr);
				}

	free(a);
	free(b);
	free(c);
}

int main(int argc, char *argv[])
{
	/* library initialization */
	aml_init(&argc, &argv);

	test_reshape_discontiguous();
	test_reshape_strided();

	aml_finalize();
	return 0;
}
