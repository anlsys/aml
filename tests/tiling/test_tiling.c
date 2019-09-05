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
	size_t expected_dims_row[3] = {3, 1, 2};

        int l = 0;
	for(size_t i = 0; i < 9; i++)
	for(size_t j = 0; j < 10; j++)
	for(size_t k = 0; k < 8; k++, l++) {
		memory[i][j][k] = l;
		memoryres[i][j][k] = 0;
	}

	struct aml_layout *a, *ares;
	aml_layout_native_create(&a, AML_TYPE_LAYOUT_COLUMN_ORDER,
				  (void *)memory, sizeof(int), 3, dims_col,
				  stride, dims_col);
	aml_layout_native_create(&ares, AML_TYPE_LAYOUT_ROW_ORDER,
				  (void *)memoryres, sizeof(int), 3, dims_row,
				  stride, dims_row);


	struct aml_tiling *t, *tres;
	aml_tiling_resize_create(&t, AML_TYPE_TILING_COLUMN_ORDER,
				     a, 3, dims_tile_col);
	aml_tiling_resize_create(&tres, AML_TYPE_TILING_ROW_ORDER,
				     ares, 3, dims_tile_row);


	for(size_t i = 0; i < expected_dims_col[2]; i++)
	for(size_t j = 0; j < expected_dims_col[1]; j++)
	for(size_t k = 0; k < expected_dims_col[0]; k++) {
		struct aml_layout *b, *bres;
		b = aml_tiling_index(t, k, j, i);
		bres = aml_tiling_index(tres, i, j, k);
		aml_copy_layout_generic(bres, b);
		free(b);
		free(bres);
	}
	assert(memcmp(memory, memoryres, 8 * 10 * 9 *sizeof(int)) == 0);

	free(a);
	free(ares);
	free(t);
	free(tres);

	aml_layout_native_create(&a, AML_TYPE_LAYOUT_COLUMN_ORDER,
				  (void *)memory, sizeof(int), 3, dims_col,
				  stride, dims_col);
	aml_layout_native_create(&ares, AML_TYPE_LAYOUT_ROW_ORDER,
				  (void *)memoryres, sizeof(int), 3, dims_row,
				  stride, dims_row);


	aml_tiling_resize_create(&t, AML_TYPE_TILING_ROW_ORDER,
				     a, 3, dims_tile_row);
	aml_tiling_resize_create(&tres, AML_TYPE_TILING_COLUMN_ORDER,
				     ares, 3, dims_tile_col);

	for(size_t i = 0; i < 9; i++)
	for(size_t j = 0; j < 10; j++)
	for(size_t k = 0; k < 8; k++, l++)
		memoryres[i][j][k] = 0.0;

	for(size_t i = 0; i < expected_dims_col[2]; i++)
	for(size_t j = 0; j < expected_dims_col[1]; j++)
	for(size_t k = 0; k < expected_dims_col[0]; k++) {
		struct aml_layout *b, *bres;
		b = aml_tiling_index(t, i, j, k);
		bres = aml_tiling_index(tres, k, j, i);
		aml_copy_layout_generic(bres, b);
		free(b);
		free(bres);
	}
	assert(memcmp(memory, memoryres, 8 * 10 * 9 *sizeof(int)) == 0);

	free(a);
	free(ares);
	free(t);
	free(tres);

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
	for(size_t i = 0; i < 9; i++)
	for(size_t j = 0; j < 10; j++)
	for(size_t k = 0; k < 8; k++, l++) {
		memory[i][j][k] = l;
		memoryres[i][j][k] = 0.0;
	}

	struct aml_layout *a, *ares;
	aml_layout_native_create(&a, AML_TYPE_LAYOUT_COLUMN_ORDER,
				  (void *)memory, sizeof(int), 3, dims_col,
				  stride, dims_col);
	aml_layout_native_create(&ares, AML_TYPE_LAYOUT_COLUMN_ORDER,
				  (void *)memoryres, sizeof(int), 3, dims_col,
				  stride, dims_col);


	struct aml_tiling *t, *tres;
	aml_tiling_resize_create(&t, AML_TYPE_TILING_COLUMN_ORDER,
				     a, 3, dims_tile_col);
	aml_tiling_resize_create(&tres, AML_TYPE_TILING_COLUMN_ORDER,
				     ares, 3, dims_tile_col);


	assert(aml_tiling_order(t) == AML_TYPE_TILING_COLUMN_ORDER);
	assert(aml_tiling_ndims(t) == 3);

        size_t dims[3];
	aml_tiling_tile_adims(t, dims);
	assert(memcmp(dims, dims_tile_col, 3*sizeof(size_t)) == 0);
	aml_tiling_dims(t, dims);
	assert(memcmp(dims, expected_dims_col, 3*sizeof(size_t)) == 0);

	for(size_t i = 0; i < expected_dims_col[2]; i++)
	for(size_t j = 0; j < expected_dims_col[1]; j++)
	for(size_t k = 0; k < expected_dims_col[0]; k++) {
		struct aml_layout *b, *bres;
		b = aml_tiling_index(t, k, j, i);
		bres = aml_tiling_index(tres, k, j, i);
		aml_copy_layout_generic(bres, b);
		free(b);
		free(bres);
	}
	assert(memcmp(memory, memoryres, 8 * 10 * 9 *sizeof(int)) == 0);

	free(a);
	free(ares);
	free(t);
	free(tres);

	aml_layout_native_create(&a, AML_TYPE_LAYOUT_ROW_ORDER,
				  (void *)memory, sizeof(int), 3, dims_row,
				  stride, dims_row);
	aml_layout_native_create(&ares, AML_TYPE_LAYOUT_ROW_ORDER,
				  (void *)memoryres, sizeof(int), 3, dims_row,
				  stride, dims_row);


	aml_tiling_resize_create(&t, AML_TYPE_TILING_ROW_ORDER,
				     a, 3, dims_tile_row);
	aml_tiling_resize_create(&tres, AML_TYPE_TILING_ROW_ORDER,
				     ares, 3, dims_tile_row);

	assert(aml_tiling_order(t) == AML_TYPE_TILING_ROW_ORDER);
	assert(aml_tiling_ndims(t) == 3);

	aml_tiling_tile_adims(t, dims);
	assert(memcmp(dims, dims_tile_row, 3*sizeof(size_t)) == 0);
	aml_tiling_dims(t, dims);
	assert(memcmp(dims, expected_dims_row, 3*sizeof(size_t)) == 0);

	for(size_t i = 0; i < 9; i++)
	for(size_t j = 0; j < 10; j++)
	for(size_t k = 0; k < 8; k++, l++)
		memoryres[i][j][k] = 0.0;

	for(size_t i = 0; i < expected_dims_col[2]; i++)
	for(size_t j = 0; j < expected_dims_col[1]; j++)
	for(size_t k = 0; k < expected_dims_col[0]; k++) {
		struct aml_layout *b, *bres;
		b = aml_tiling_index(t, i, j, k);
		bres = aml_tiling_index(tres, i, j, k);
		aml_copy_layout_generic(bres, b);
		free(b);
		free(bres);
	}
	assert(memcmp(memory, memoryres, 8 * 10 * 9 *sizeof(int)) == 0);

	free(a);
	free(ares);
	free(t);
	free(tres);

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
	for(size_t i = 0; i < 8; i++)
	for(size_t j = 0; j < 10; j++)
	for(size_t k = 0; k < 7; k++, l++) {
		memory[i][j][k] = l;
		memoryres[i][j][k] = 0.0;
	}

	struct aml_layout *a, *ares;
	aml_layout_native_create(&a, AML_TYPE_LAYOUT_COLUMN_ORDER,
				  (void *)memory, sizeof(int), 3, dims_col,
				  stride, dims_col);
	aml_layout_native_create(&ares, AML_TYPE_LAYOUT_COLUMN_ORDER,
				  (void *)memoryres, sizeof(int), 3, dims_col,
				  stride, dims_col);


	struct aml_tiling *t, *tres;
	aml_tiling_resize_create(&t, AML_TYPE_TILING_COLUMN_ORDER,
				     a, 3, dims_tile_col);
	aml_tiling_resize_create(&tres, AML_TYPE_TILING_COLUMN_ORDER,
				     ares, 3, dims_tile_col);


	assert(aml_tiling_order(t) == AML_TYPE_TILING_COLUMN_ORDER);
	assert(aml_tiling_ndims(t) == 3);

        size_t dims[3];
	aml_tiling_tile_adims(t, dims);
	assert(memcmp(dims, dims_tile_col, 3*sizeof(size_t)) == 0);
	aml_tiling_dims(t, dims);
	assert(memcmp(dims, expected_dims_col, 3*sizeof(size_t)) == 0);

	for(size_t i = 0; i < expected_dims_col[2]; i++)
	for(size_t j = 0; j < expected_dims_col[1]; j++)
	for(size_t k = 0; k < expected_dims_col[0]; k++) {
		struct aml_layout *b, *bres;
		b = aml_tiling_index(t, k, j, i);
		bres = aml_tiling_index(tres, k, j, i);
		aml_copy_layout_generic(bres, b);
		free(b);
		free(bres);
	}
	assert(memcmp(memory, memoryres, 7 * 10 * 8 *sizeof(int)) == 0);

	free(a);
	free(ares);
	free(t);
	free(tres);

	aml_layout_native_create(&a, AML_TYPE_LAYOUT_ROW_ORDER,
				  (void *)memory, sizeof(int), 3, dims_row,
				  stride, dims_row);
	aml_layout_native_create(&ares, AML_TYPE_LAYOUT_ROW_ORDER,
				  (void *)memoryres, sizeof(int), 3, dims_row,
				  stride, dims_row);

	aml_tiling_resize_create(&t, AML_TYPE_TILING_ROW_ORDER,
				     a, 3, dims_tile_row);
	aml_tiling_resize_create(&tres, AML_TYPE_TILING_ROW_ORDER,
				     ares, 3, dims_tile_row);

	assert(aml_tiling_order(t) == AML_TYPE_TILING_ROW_ORDER);
	assert(aml_tiling_ndims(t) == 3);

	aml_tiling_tile_adims(t, dims);
	assert(memcmp(dims, dims_tile_row, 3*sizeof(size_t)) == 0);
	aml_tiling_dims(t, dims);
	assert(memcmp(dims, expected_dims_row, 3*sizeof(size_t)) == 0);

	for(size_t i = 0; i < 8; i++)
	for(size_t j = 0; j < 10; j++)
	for(size_t k = 0; k < 7; k++, l++)
		memoryres[i][j][k] = 0.0;

	for(size_t i = 0; i < expected_dims_col[2]; i++)
	for(size_t j = 0; j < expected_dims_col[1]; j++)
	for(size_t k = 0; k < expected_dims_col[0]; k++) {
		struct aml_layout *b, *bres;
		b = aml_tiling_index(t, i, j, k);
		bres = aml_tiling_index(tres, i, j, k);
		aml_copy_layout_generic(bres, b);
		free(b);
		free(bres);
	}
	assert(memcmp(memory, memoryres, 7 * 10 * 8 *sizeof(int)) == 0);

	free(a);
	free(ares);
	free(t);
	free(tres);

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

