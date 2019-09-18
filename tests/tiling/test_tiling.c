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
#include "aml/layout/dense.h"
#include "aml/layout/native.h"
#include "aml/tiling/resize.h"
#include <assert.h>

void test_tiling_even_mixed(void)
{
	int memory[9][10][8];
	int memoryres[9][10][8];
	size_t dims_col[3] = {8, 10, 9};
	size_t dims_row[3] = {9, 10, 8};

	size_t stride[3] = {1, 1, 1};

	size_t dims_tile_col[3] = {4, 10, 3};
	size_t dims_tile_row[3] = {3, 10, 4};

	size_t expected_dims_col[3] = {2, 1, 3};

	int l = 0;

	for (size_t i = 0; i < 9; i++)
		for (size_t j = 0; j < 10; j++)
			for (size_t k = 0; k < 8; k++, l++) {
				memory[i][j][k] = l;
				memoryres[i][j][k] = 0;
			}

	struct aml_layout *a, *ares;

	aml_layout_dense_create(&a, (void *)memory,
				  AML_LAYOUT_ORDER_COLUMN_MAJOR,
				  sizeof(int), 3, dims_col,
				  stride, dims_col);
	aml_layout_dense_create(&ares, (void *)memoryres,
				AML_LAYOUT_ORDER_ROW_MAJOR,
				sizeof(int), 3, dims_row,
				stride, dims_row);


	struct aml_tiling *t, *tres;

	aml_tiling_resize_create(&t, AML_TILING_ORDER_COLUMN_MAJOR,
				     a, 3, dims_tile_col);
	aml_tiling_resize_create(&tres, AML_TILING_ORDER_ROW_MAJOR,
				     ares, 3, dims_tile_row);


	for (size_t i = 0; i < expected_dims_col[2]; i++)
		for (size_t j = 0; j < expected_dims_col[1]; j++)
			for (size_t k = 0; k < expected_dims_col[0]; k++) {
				struct aml_layout *b, *bres;

				b = aml_tiling_index(t, (size_t[]){k, j, i});
				bres = aml_tiling_index(tres,
							(size_t[]){i, j, k});
				aml_copy_layout_generic(bres, b, NULL);
				free(b);
				free(bres);
			}
	assert(memcmp(memory, memoryres, 8 * 10 * 9 * sizeof(int)) == 0);

	aml_layout_dense_destroy(&a);
	aml_layout_dense_destroy(&ares);
	aml_tiling_resize_destroy(&t);
	aml_tiling_resize_destroy(&tres);

	aml_layout_dense_create(&a, memory, AML_LAYOUT_ORDER_COLUMN_MAJOR,
				sizeof(int), 3, dims_col,
				  stride, dims_col);
	aml_layout_dense_create(&ares, memoryres, AML_LAYOUT_ORDER_ROW_MAJOR,
				sizeof(int), 3, dims_row,
				  stride, dims_row);


	aml_tiling_resize_create(&t, AML_TILING_ORDER_ROW_MAJOR,
				     a, 3, dims_tile_row);
	aml_tiling_resize_create(&tres, AML_TILING_ORDER_COLUMN_MAJOR,
				     ares, 3, dims_tile_col);

	for (size_t i = 0; i < 9; i++)
		for (size_t j = 0; j < 10; j++)
			for (size_t k = 0; k < 8; k++, l++)
				memoryres[i][j][k] = 0.0;

	for (size_t i = 0; i < expected_dims_col[2]; i++)
		for (size_t j = 0; j < expected_dims_col[1]; j++)
			for (size_t k = 0; k < expected_dims_col[0]; k++) {
				struct aml_layout *b, *bres;

				b = aml_tiling_index(t, (size_t[]){i, j, k});
				bres = aml_tiling_index(tres,
							(size_t[]){k, j, i});
				aml_copy_layout_generic(bres, b, NULL);
				free(b);
				free(bres);
			}
	assert(memcmp(memory, memoryres, 8 * 10 * 9 * sizeof(int)) == 0);

	aml_layout_dense_destroy(&a);
	aml_layout_dense_destroy(&ares);
	aml_tiling_resize_destroy(&t);
	aml_tiling_resize_destroy(&tres);

}

void test_tiling_even(void)
{
	int memory[9][10][8];
	int memoryres[9][10][8];
	size_t dims_col[3] = {8, 10, 9};
	size_t dims_row[3] = {9, 10, 8};

	size_t stride[3] = {1, 1, 1};

	size_t dims_tile_col[3] = {4, 10, 3};
	size_t dims_tile_row[3] = {3, 10, 4};

	size_t expected_dims_col[3] = {2, 1, 3};
	size_t expected_dims_row[3] = {3, 1, 2};

	int l = 0;

	for (size_t i = 0; i < 9; i++)
		for (size_t j = 0; j < 10; j++)
			for (size_t k = 0; k < 8; k++, l++) {
				memory[i][j][k] = l;
				memoryres[i][j][k] = 0.0;
			}

	struct aml_layout *a, *ares;

	aml_layout_dense_create(&a, memory, AML_LAYOUT_ORDER_COLUMN_MAJOR,
				sizeof(int), 3, dims_col,
				stride, dims_col);
	aml_layout_dense_create(&ares, memoryres, AML_LAYOUT_ORDER_COLUMN_MAJOR,
				sizeof(int), 3, dims_col,
				stride, dims_col);


	struct aml_tiling *t, *tres;

	aml_tiling_resize_create(&t, AML_TILING_ORDER_COLUMN_MAJOR,
				     a, 3, dims_tile_col);
	aml_tiling_resize_create(&tres, AML_TILING_ORDER_COLUMN_MAJOR,
				     ares, 3, dims_tile_col);


	assert(aml_tiling_order(t) == AML_TILING_ORDER_COLUMN_MAJOR);
	assert(aml_tiling_ndims(t) == 3);

	size_t dims[3];

	aml_tiling_tile_dims(t, dims);
	assert(memcmp(dims, dims_tile_col, 3*sizeof(size_t)) == 0);
	aml_tiling_dims(t, dims);
	assert(memcmp(dims, expected_dims_col, 3*sizeof(size_t)) == 0);

	for (size_t i = 0; i < expected_dims_col[2]; i++)
		for (size_t j = 0; j < expected_dims_col[1]; j++)
			for (size_t k = 0; k < expected_dims_col[0]; k++) {
				struct aml_layout *b, *bres;

				b = aml_tiling_index(t, (size_t[]){k, j, i});
				bres = aml_tiling_index(tres,
							(size_t[]){k, j, i});
				aml_copy_layout_generic(bres, b, NULL);
				free(b);
				free(bres);
			}
	assert(memcmp(memory, memoryres, 8 * 10 * 9 * sizeof(int)) == 0);

	aml_layout_dense_destroy(&a);
	aml_layout_dense_destroy(&ares);
	aml_tiling_resize_destroy(&t);
	aml_tiling_resize_destroy(&tres);

	aml_layout_dense_create(&a, memory, AML_LAYOUT_ORDER_ROW_MAJOR,
				sizeof(int), 3, dims_row,
				stride, dims_row);
	aml_layout_dense_create(&ares, memoryres, AML_LAYOUT_ORDER_ROW_MAJOR,
				sizeof(int), 3, dims_row,
				stride, dims_row);


	aml_tiling_resize_create(&t, AML_TILING_ORDER_ROW_MAJOR,
				     a, 3, dims_tile_row);
	aml_tiling_resize_create(&tres, AML_TILING_ORDER_ROW_MAJOR,
				     ares, 3, dims_tile_row);

	assert(aml_tiling_order(t) == AML_TILING_ORDER_ROW_MAJOR);
	assert(aml_tiling_ndims(t) == 3);

	aml_tiling_tile_dims(t, dims);
	assert(memcmp(dims, dims_tile_row, 3*sizeof(size_t)) == 0);
	aml_tiling_dims(t, dims);
	assert(memcmp(dims, expected_dims_row, 3*sizeof(size_t)) == 0);

	for (size_t i = 0; i < 9; i++)
		for (size_t j = 0; j < 10; j++)
			for (size_t k = 0; k < 8; k++, l++)
				memoryres[i][j][k] = 0.0;

	for (size_t i = 0; i < expected_dims_col[2]; i++)
		for (size_t j = 0; j < expected_dims_col[1]; j++)
			for (size_t k = 0; k < expected_dims_col[0]; k++) {
				struct aml_layout *b, *bres;

				b = aml_tiling_index(t, (size_t[]){i, j, k});
				bres = aml_tiling_index(tres,
							(size_t[]){i, j, k});
				assert(b != NULL && bres != NULL);
				aml_copy_layout_generic(bres, b, NULL);
				free(b);
				free(bres);
			}
	assert(memcmp(memory, memoryres, 8 * 10 * 9 * sizeof(int)) == 0);

	aml_layout_dense_destroy(&a);
	aml_layout_dense_destroy(&ares);
	aml_tiling_resize_destroy(&t);
	aml_tiling_resize_destroy(&tres);

}

void test_tiling_uneven(void)
{

	int memory[8][10][7];
	int memoryres[8][10][7];
	size_t dims_col[3] = {7, 10, 8};
	size_t dims_row[3] = {8, 10, 7};

	size_t stride[3] = {1, 1, 1};

	size_t dims_tile_col[3] = {4, 10, 3};
	size_t dims_tile_row[3] = {3, 10, 4};

	size_t expected_dims_col[3] = {2, 1, 3};
	size_t expected_dims_row[3] = {3, 1, 2};

	int l = 0;

	for (size_t i = 0; i < 8; i++)
		for (size_t j = 0; j < 10; j++)
			for (size_t k = 0; k < 7; k++, l++) {
				memory[i][j][k] = l;
				memoryres[i][j][k] = 0.0;
			}

	struct aml_layout *a, *ares;

	aml_layout_dense_create(&a, memory, AML_LAYOUT_ORDER_COLUMN_MAJOR,
				sizeof(int), 3, dims_col,
				stride, dims_col);
	aml_layout_dense_create(&ares, memoryres, AML_LAYOUT_ORDER_COLUMN_MAJOR,
				sizeof(int), 3, dims_col,
				stride, dims_col);


	struct aml_tiling *t, *tres;

	aml_tiling_resize_create(&t, AML_TILING_ORDER_COLUMN_MAJOR,
				     a, 3, dims_tile_col);
	aml_tiling_resize_create(&tres, AML_TILING_ORDER_COLUMN_MAJOR,
				     ares, 3, dims_tile_col);


	assert(aml_tiling_order(t) == AML_TILING_ORDER_COLUMN_MAJOR);
	assert(aml_tiling_ndims(t) == 3);

	size_t dims[3];

	aml_tiling_tile_dims(t, dims);
	assert(memcmp(dims, dims_tile_col, 3*sizeof(size_t)) == 0);
	aml_tiling_dims(t, dims);
	assert(memcmp(dims, expected_dims_col, 3*sizeof(size_t)) == 0);

	for (size_t i = 0; i < expected_dims_col[2]; i++)
		for (size_t j = 0; j < expected_dims_col[1]; j++)
			for (size_t k = 0; k < expected_dims_col[0]; k++) {
				struct aml_layout *b, *bres;

				b = aml_tiling_index(t, (size_t[]){k, j, i});
				bres = aml_tiling_index(tres,
							(size_t[]){k, j, i});
				aml_copy_layout_generic(bres, b, NULL);
				free(b);
				free(bres);
			}
	assert(memcmp(memory, memoryres, 7 * 10 * 8 * sizeof(int)) == 0);

	aml_layout_dense_destroy(&a);
	aml_layout_dense_destroy(&ares);
	aml_tiling_resize_destroy(&t);
	aml_tiling_resize_destroy(&tres);

	aml_layout_dense_create(&a, memory, AML_LAYOUT_ORDER_ROW_MAJOR,
				sizeof(int), 3, dims_row,
				  stride, dims_row);
	aml_layout_dense_create(&ares, memoryres, AML_LAYOUT_ORDER_ROW_MAJOR,
				sizeof(int), 3, dims_row,
				  stride, dims_row);

	aml_tiling_resize_create(&t, AML_TILING_ORDER_ROW_MAJOR,
				     a, 3, dims_tile_row);
	aml_tiling_resize_create(&tres, AML_TILING_ORDER_ROW_MAJOR,
				     ares, 3, dims_tile_row);

	assert(aml_tiling_order(t) == AML_TILING_ORDER_ROW_MAJOR);
	assert(aml_tiling_ndims(t) == 3);

	aml_tiling_tile_dims(t, dims);
	assert(memcmp(dims, dims_tile_row, 3*sizeof(size_t)) == 0);
	aml_tiling_dims(t, dims);
	assert(memcmp(dims, expected_dims_row, 3*sizeof(size_t)) == 0);

	for (size_t i = 0; i < 8; i++)
		for (size_t j = 0; j < 10; j++)
			for (size_t k = 0; k < 7; k++, l++) {
				memory[i][j][k] = l;
				memoryres[i][j][k] = 0.0;
			}

	for (size_t i = 0; i < expected_dims_col[2]; i++)
		for (size_t j = 0; j < expected_dims_col[1]; j++)
			for (size_t k = 0; k < expected_dims_col[0]; k++) {
				struct aml_layout *b, *bres;

				b = aml_tiling_index(t, (size_t[]){i, j, k});
				bres = aml_tiling_index(tres,
							(size_t[]){i, j, k});
				aml_copy_layout_generic(bres, b, NULL);
				free(b);
				free(bres);
			}
	assert(memcmp(memory, memoryres, 7 * 10 * 8 * sizeof(int)) == 0);

	aml_layout_dense_destroy(&a);
	aml_layout_dense_destroy(&ares);
	aml_tiling_resize_destroy(&t);
	aml_tiling_resize_destroy(&tres);
}

int main(int argc, char *argv[])
{
	/* library initialization */
	aml_init(&argc, &argv);

	test_tiling_even();
	test_tiling_uneven();
	test_tiling_even_mixed();

	return 0;
}

