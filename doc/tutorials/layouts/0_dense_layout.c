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

void print_layout(struct aml_layout *layout)
{
	size_t ndims = aml_layout_ndims(layout);
	size_t dims[ndims];
	size_t coords[ndims];
	double *a;

	aml_layout_dims(layout, dims);
	for (size_t i = 0; i < dims[0]; i++) {
		for (size_t j = 0; j < dims[1]; j++) {
			coords[0] = i;
			coords[1] = j;
			a = aml_layout_deref(layout, coords);
			fprintf(stderr, "%f ", *a);
		}
		fprintf(stderr, "\n");
	}
}

int main(int argc, char **argv)
{
	if (aml_init(&argc, &argv) != AML_SUCCESS)
		return 1;

	size_t x = 4, y = 5;
	double *array;
	struct aml_area *area = &aml_area_linux;
	struct aml_layout *layout_c, *layout_f;

	// Allocationg our matrix through an area
	 array = (double *)aml_area_mmap(area, sizeof(double) * x * y, NULL);

	// Initializing our matrix
	for (size_t i = 0; i < x*y; i++)
		array[i] = i;

	fprintf(stderr, "Creating layouts...\n");

	// Layout ordered columns first
	size_t dims[2] = { x, y };

	aml_layout_dense_create(&layout_c, array, AML_LAYOUT_ORDER_C,
				sizeof(double), 2, dims, NULL, NULL);

	// Layout ordered rows first, on the same area
	aml_layout_dense_create(&layout_f, array, AML_LAYOUT_ORDER_FORTRAN,
				sizeof(double), 2, dims, NULL, NULL);


	assert(layout_c != NULL && layout_f != NULL);

	/* Check the difference with aml_layout_order */
	aml_layout_fprintf(stderr, "layout_c", layout_c);
	print_layout(layout_c);
	aml_layout_fprintf(stderr, "layout_f", layout_f);
	print_layout(layout_f);

	aml_layout_dense_destroy(&layout_c);
	aml_layout_dense_destroy(&layout_f);
	aml_area_munmap(area, array, sizeof(double) * x * y);

	aml_finalize();
	return 0;
}
