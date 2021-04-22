/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include <aml.h>
#include "aml/area/linux.h"
#include "aml/layout/dense.h"
#include "aml/tiling/resize.h"
#include <stdio.h>

void init_matrix(double *mat, size_t rows, size_t cols, double scal)
{
	for (size_t i = 0; i < rows; i++)
		for (size_t j = 0; j < cols; j++)
			mat[j * rows + i] = scal;
}

void print_matrix(double *mat, size_t rows, size_t cols)
{
	for (size_t i = 0; i < rows; i++) {
		for (size_t j = 0; j < cols; j++)
			fprintf(stderr, "%f ", mat[j * rows + i]);
		fprintf(stderr, "\n");
	}
	fprintf(stderr, "\n");
}

void matrix_multiplication(int m, int n, int k,
			   double *a, double *b, double *c)
{
	int i, j, l;

	for (i = 0; i < m; i++)
		for (j = 0; j < n; j++)
			for (l = 0; l < k; l++)
				c[j * m + i] += a[l * m + i] * b[j * k + l];
}

void dgemm_layout(size_t mt, size_t nt, size_t kt,
		  struct aml_layout *ltile_a,
		  struct aml_layout *ltile_b,
		  struct aml_layout *ltile_c)
{
	for (size_t ii = 0; ii < mt; ii++) {
		for (size_t jj = 0; jj < nt; jj++) {
			double *ct;

			ct = aml_layout_deref(ltile_c, (size_t[]){ii, jj});
			for (size_t ll = 0; ll < kt; ll++) {
				double *at, *bt;

				at = aml_layout_deref(ltile_a,
						      (size_t[]){ii, ll});
				bt = aml_layout_deref(ltile_b,
						      (size_t[]){ll, jj});
				*ct += *at * *bt;
			}
		}
	}
}

int dgemm_tiled(struct aml_tiling *tiling_a, struct aml_tiling *tiling_b,
		struct aml_tiling *tiling_c)
{
	size_t ndims = aml_tiling_ndims(tiling_a);
	size_t tile_a_dims[ndims];
	size_t tile_b_dims[ndims];
	size_t mt, nt, kt;
	size_t tiling_a_dims[ndims];
	size_t tiling_b_dims[ndims];
	size_t tiling_c_dims[ndims];

	aml_tiling_tile_dims(tiling_a, NULL, tile_a_dims);
	aml_tiling_tile_dims(tiling_b, NULL, tile_b_dims);
	mt = tile_a_dims[0];
	kt = tile_a_dims[1];
	nt = tile_b_dims[1];
	aml_tiling_dims(tiling_a, tiling_a_dims);
	aml_tiling_dims(tiling_b, tiling_b_dims);
	aml_tiling_dims(tiling_c, tiling_c_dims);

	// Going through the tiling tile by tile
	for (size_t i = 0; i < tiling_c_dims[0]; i++) {
		for (size_t j = 0; j < tiling_c_dims[1]; j++) {
			struct aml_layout *ltile_c;

			ltile_c = aml_tiling_index(tiling_c, (size_t[]){i, j});
			for (size_t l = 0; l < tiling_a_dims[1]; l++) {
				struct aml_layout *ltile_a, *ltile_b;

				ltile_a = aml_tiling_index(tiling_a,
						       (size_t[]){i, l});
				ltile_b = aml_tiling_index(tiling_b,
						       (size_t[]){l, j});
				dgemm_layout(mt, nt, kt, ltile_a, ltile_b,
					     ltile_c);
				aml_layout_destroy(&ltile_a);
				aml_layout_destroy(&ltile_b);
			}
			aml_layout_destroy(&ltile_c);
		}
	}
	return AML_SUCCESS;
}

int main(int argc, char **argv)
{
	if (aml_init(&argc, &argv) != AML_SUCCESS)
		return 1;

	size_t m = 6, n = 12, k = 9;
	double *a, *b, *c, *c_ref;

	// Allocationg our matrices
	a = malloc(sizeof(double) * m * k);
	b = malloc(sizeof(double) * k * n);
	c = malloc(sizeof(double) * m * n);
	c_ref = malloc(sizeof(double) * m * n);
	assert(a != NULL && b != NULL && c != NULL && c_ref != NULL);

	// Initializing our matrices, all of a is 1, all of b is 2
	init_matrix(a, m, k, 1.0);
	init_matrix(b, k, n, 2.0);
	init_matrix(c, m, n, 0.0);
	init_matrix(c_ref, m, n, 0.0);

	// Calculating our reference result
	matrix_multiplication(m, n, k, a, b, c_ref);
	print_matrix(c_ref, m, n);

	fprintf(stderr, "Creating layouts...\n");

	struct aml_layout *layout_a, *layout_b, *layout_c;
	size_t dims_a[2] = { m, k };
	size_t dims_b[2] = { k, n };
	size_t dims_c[2] = { m, n };

	assert(!aml_layout_dense_create(&layout_a, a,
					AML_LAYOUT_ORDER_C,
					sizeof(double), 2, dims_a, NULL,
					NULL));
	assert(!aml_layout_dense_create(&layout_b, b,
					AML_LAYOUT_ORDER_C,
					sizeof(double), 2, dims_b, NULL,
					NULL));
	assert(!aml_layout_dense_create(&layout_c, c,
					AML_LAYOUT_ORDER_C,
					sizeof(double), 2, dims_c, NULL,
					NULL));

	// We divide each matrix in 9 blocks
	fprintf(stderr, "Creating tilings...\n");

	struct aml_tiling *tiling_a, *tiling_b, *tiling_c;
	size_t tile_a_dims[2] = { 2, 3 };
	size_t tile_b_dims[2] = { 3, 4 };
	size_t tile_c_dims[2] = { 2, 4 };

	assert(!aml_tiling_resize_create(&tiling_a,
					 AML_TILING_ORDER_C,
					 layout_a, 2, tile_a_dims));
	assert(!aml_tiling_resize_create(&tiling_b,
					 AML_TILING_ORDER_C,
					 layout_b, 2, tile_b_dims));
	assert(!aml_tiling_resize_create(&tiling_c,
					 AML_TILING_ORDER_C,
					 layout_c, 2, tile_c_dims));

	// Do the matrix multiplication
	assert(!dgemm_tiled(tiling_a, tiling_b, tiling_c));
	print_matrix(c, m, n);

	/* Destroy everything */
	aml_tiling_resize_destroy(&tiling_a);
	aml_tiling_resize_destroy(&tiling_b);
	aml_tiling_resize_destroy(&tiling_c);
	aml_layout_destroy(&layout_a);
	aml_layout_destroy(&layout_b);
	aml_layout_destroy(&layout_c);
	free(a);
	free(b);
	free(c);
	free(c_ref);

	aml_finalize();

	return 0;
}
