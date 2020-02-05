/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include "aml.h"
#include "aml/area/linux.h"
#include "aml/layout/dense.h"
#include <stdio.h>

void print_matrix(double *mat, size_t rows, size_t cols)
{
	for (size_t i = 0; i < rows; i++) {
		for (size_t j = 0; j < cols; j++)
			fprintf(stderr, "%f ", mat[i * rows + j]);
		fprintf(stderr, "\n");
	}
	fprintf(stderr, "\n");
}

/* Initialize a given layout with arbitrary values.
 * Use of
 *	aml_layout_ndims
 *	aml_layout_element_size
 *	aml_layout_dims
 *	aml_layout_deref
 */
int fill_layout(struct aml_layout *layout)
{
	int err;
	size_t element_size;
	size_t ndims = aml_layout_ndims(layout);
	size_t dims[ndims];
	size_t coords[ndims];
	double *a;
	double count = 1;

	element_size = aml_layout_element_size(layout);
	assert(element_size != 0);

	err = aml_layout_dims(layout, dims);
	if (err != AML_SUCCESS)
		return err;

	// Going through the each dimension of the layout
	for (size_t i = 0; i < dims[0]; i++) {
		coords[0] = i;
		for (size_t j = 0; j < dims[1]; j++) {
			coords[1] = j;
			a = aml_layout_deref(layout, coords);
			*a = count;
			count++;
		}
	}

	return AML_SUCCESS;
}

int main(int argc, char **argv)
{
	if (aml_init(&argc, &argv) != AML_SUCCESS)
		return 1;

	size_t x = 4, y = 5;
	double *mat;
	struct aml_area *area = &aml_area_linux;
	struct aml_layout *layout_c, *layout_r;

	// Allocationg our matrix through an area
	mat = (double *)aml_area_mmap(area, sizeof(double) * x * y, NULL);

	// Initializing our matrix
	for (size_t i = 0; i < x; i++)
		for (size_t j = 0; j < y; j++)
			mat[i * x + j] = 0;

	fprintf(stderr, "Creating layouts...\n");

	// Layout ordered columns first
	size_t dims_col[3] = { x, y };

	aml_layout_dense_create(&layout_c, mat, AML_LAYOUT_ORDER_COLUMN_MAJOR,
				sizeof(double), 2, dims_col, NULL, NULL);

	// Layout ordered rows first, on the same area
	size_t dims_row[3] = { y, x };

	aml_layout_dense_create(&layout_r, mat, AML_LAYOUT_ORDER_ROW_MAJOR,
				sizeof(double), 2, dims_row, NULL, NULL);

	assert(layout_c != NULL && layout_r != NULL);

	/* Check the difference with aml_layout_order */
	fprintf(stderr, "The first layout has order %d: column major, ",
		aml_layout_order(layout_c));
	fprintf(stderr, "the second layout has order %d: row major.\n",
		aml_layout_order(layout_r));

	/* Filling the matrix with different layouts */

	fill_layout(layout_c);
	print_matrix(mat, x, y);
	fill_layout(layout_r);
	print_matrix(mat, x, y);

	aml_layout_dense_destroy(&layout_c);
	aml_layout_dense_destroy(&layout_r);
	aml_area_munmap(area, mat, sizeof(double) * x * y);

	aml_finalize();

	return 0;
}
