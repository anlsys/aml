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

int print_layout(struct aml_layout *layout)
{
	size_t ndims = aml_layout_ndims(layout);
	size_t dims[ndims];
	size_t coords[ndims];
	double *a;

	aml_layout_dims(layout, dims);

	// Going through the each dimension of the layout
	for (size_t i = 0; i < dims[2]; i++) {
		coords[2] = i;
		fprintf(stderr, "Third dimension, index %ld\n", i);
		for (size_t j = 0; j < dims[0]; j++) {
			coords[0] = j;
			for (size_t k = 0; k < dims[1]; k++) {
				coords[1] = k;
				a = aml_layout_deref(layout, coords);
				fprintf(stderr, "%f ", *a);
			}
			fprintf(stderr, "\n");
		}
	}
	fprintf(stderr, "\n");

	return AML_SUCCESS;
}

/* Change the shape of a layout
 * aml_layout_reshape
 */

/* Slice a layout
 * aml_layout_slice
 */

int main(int argc, char **argv)
{
	if (aml_init(&argc, &argv) != AML_SUCCESS)
		return 1;

	// Layout creation and initialization
	size_t x = 4, y = 5;
	double *mat;
	struct aml_area *area = &aml_area_linux;
	struct aml_layout *layout_c;

	// Allocationg our matrix through an area
	mat = (double *)aml_area_mmap(area, sizeof(double) * x * y, NULL);

	// Initializing our matrix
	// Going through the each dimension of the layout
	for (size_t i = 0; i < x; i++)
		for (size_t j = 0; j < y; j++)
			mat[i * x + j] = 0;

	fprintf(stderr, "Creating our layout...\n");

	// Layout ordered columns first
	size_t dims_col[3] = { x, y };

	aml_layout_dense_create(&layout_c, mat, AML_LAYOUT_ORDER_COLUMN_MAJOR,
				sizeof(double), 2, dims_col, NULL, NULL);

	assert(layout_c != NULL);

	/* Filling the matrix with different layouts */

	fill_layout(layout_c);
	print_matrix(mat, x, y);

	/* Let's change the shape of our layouts now */
	size_t new_dims[3] = { x/2, x/2, y };
	struct aml_layout *reshape_layout;

	assert(!aml_layout_reshape(layout_c, &reshape_layout, 3, new_dims));
	print_layout(reshape_layout);

	aml_layout_dense_destroy(&layout_c);
	aml_layout_dense_destroy(&reshape_layout);
	aml_area_munmap(area, mat, sizeof(double) * x * y);

	aml_finalize();

	return 0;
}
