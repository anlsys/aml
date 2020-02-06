/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include <aml.h>
#include "aml/area/linux.h"
#include "aml/layout/dense.h"
#include "aml/tiling/resize.h"
#include <stdio.h>

void print_matrix(double *mat, size_t rows, size_t cols)
{
	for (size_t i = 0; i < rows; i++) {
		for (size_t j = 0; j < cols; j++)
			fprintf(stderr, "%f ", mat[i * cols + j]);
		fprintf(stderr, "\n");
	}
	fprintf(stderr, "\n");
}

/* Initialize a given tiling with arbitrary values.
 * Use of
 *	aml_tiling_ndims
 *	aml_tiling_tile_dims
 *	aml_tiling_dims
 *	aml_tiling_ntiles
 *	aml_tiling_index_byid
 *
 */
int fill_tiling(struct aml_tiling *tiling)
{
	int err;
	size_t ndims = aml_tiling_ndims(tiling);
	size_t ntiles = aml_tiling_ntiles(tiling);
	size_t dims[ndims];
	size_t tile_dims[ndims];
	size_t coords[ndims];
	double *a;
	double count = 1;

	err = aml_tiling_dims(tiling, dims);
	if (err != AML_SUCCESS)
		return err;

	aml_tiling_tile_dims(tiling, tile_dims);

	// Going through the tiling tile by tile
	for (size_t i = 0; i < ntiles; i++) {
		struct aml_layout *ltile;

		// We get the layout associated with the current tile
		ltile = aml_tiling_index_byid(tiling, i);
		assert(ltile != NULL);

		// Then we fill the layout element by element
		for (size_t j = 0; j < tile_dims[0]; j++) {
			coords[0] = j;
			for (size_t k = 0; k < tile_dims[1]; k++) {
				coords[1] = k;
				a = aml_layout_deref(ltile, coords);
				*a = count;
				count++;
			}
		}

		aml_layout_dense_destroy(&ltile);
	}
	return AML_SUCCESS;
}

int main(int argc, char **argv)
{
	if (aml_init(&argc, &argv) != AML_SUCCESS)
		return 1;

	size_t x = 6, y = 9;
	double *mat;
	struct aml_area *area = &aml_area_linux;

	// Allocationg our matrix through an area
	mat = (double *)aml_area_mmap(area, sizeof(double) * x * y, NULL);
	assert(mat != NULL);

	// Initializing our matrix
	for (size_t i = 0; i < x; i++)
		for (size_t j = 0; j < y; j++)
			mat[i * x + j] = 0;

	fprintf(stderr, "Creating layouts...\n");

	// Layout ordered columns first
	struct aml_layout *layout_c, *layout_f;
	size_t dims[3] = { x, y };

	assert(!aml_layout_dense_create(&layout_c, mat,
					AML_LAYOUT_ORDER_C,
					sizeof(double), 2, dims, NULL,
					NULL));

	// Layout ordered rows first, on the same area

	assert(!aml_layout_dense_create(&layout_f, mat,
					AML_LAYOUT_ORDER_FORTRAN,
					sizeof(double), 2, dims, NULL,
					NULL));

	assert(layout_c != NULL && layout_f != NULL);

	/* Tilings, both orders */
	fprintf(stderr, "Creating tilings...\n");

	struct aml_tiling *tiling_c, *tiling_f;
	size_t tile_x = 2, tile_y = 3;

	assert(!aml_tiling_resize_create(&tiling_c,
					 AML_TILING_ORDER_C,
					 layout_c, 2,
					 (size_t[]){tile_x, tile_y}));
	assert(!aml_tiling_resize_create(&tiling_f,
					 AML_TILING_ORDER_FORTRAN,
					 layout_f, 2,
					 (size_t[]){tile_x, tile_y}));

	assert(tiling_c != NULL && tiling_f != NULL);

	/* Check the difference with aml_tiling_order */
	fprintf(stderr, "The first tiling has order %d: c, ",
		aml_tiling_order(tiling_c));
	fprintf(stderr, "the second tiling has order %d: fortran.\n",
		aml_tiling_order(tiling_f));

	/* Fill them in order */
	fprintf(stderr, "Going through the tilings...\n");
	fill_tiling(tiling_c);
	print_matrix(mat, x, y);
	fprintf(stderr, "\n");
	fill_tiling(tiling_f);
	print_matrix(mat, x, y);

	/* Destroy everything */
	aml_tiling_resize_destroy(&tiling_c);
	aml_tiling_resize_destroy(&tiling_f);
	aml_layout_dense_destroy(&layout_c);
	aml_layout_dense_destroy(&layout_f);
	aml_area_munmap(area, mat, sizeof(double) * x * y);

	aml_finalize();

	return 0;
}
