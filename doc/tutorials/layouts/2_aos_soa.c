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
#include "aml/dma/linux-seq.h"
#include <stdio.h>

struct particle {
	size_t id;
	size_t position_x;
	size_t position_y;
	double energy;
};

int main(int argc, char **argv)
{
	const size_t size_1 = (1 << 9);
	const size_t size_2 = (1 << 7);
	size_t size = size_1 * size_2;

	if (aml_init(&argc, &argv) != AML_SUCCESS)
		return 1;

	struct particle *particles;

	// Allocate memory for the array of struct through an area
	struct aml_area *area = &aml_area_linux;

	particles = (struct particle *)
		aml_area_mmap(area, sizeof(struct particle) * size, NULL);

	// Initializing our array of particles
	for (size_t i = 0; i < size; i++) {
		particles[i].id = i;
		particles[i].position_x = i;
		particles[i].position_y = i;
		particles[i].energy = (double) 2*i;
	}
	assert(sizeof(struct particle) == sizeof(size_t) * 4);

	fprintf(stderr, "Creating layouts...\n");
	struct aml_layout *lay_part, *layout_elements, *new_layout;

	// We start with a straighforward layout
	assert(!aml_layout_dense_create(&lay_part, particles,
					AML_LAYOUT_ORDER_COLUMN_MAJOR, //FORTRAN
					sizeof(struct particle), 2,
					(size_t[]){size_1, size_2}, NULL,
					NULL));

	assert(lay_part != NULL);

	// We need a finer layout
	assert(!aml_layout_dense_create(&layout_elements, particles,
					AML_LAYOUT_ORDER_COLUMN_MAJOR, //FORTRAN
					sizeof(size_t), 3,
					(size_t[]){4, size_1, size_2},
					NULL, NULL));

	// Let's take a look at it
	for (size_t i = 0; i < 10; i++) {
		fprintf(stderr, "%ld ",
			*(size_t *)aml_layout_deref(layout_elements,
						    (size_t[]){0, i, 0}));
		fprintf(stderr, "%ld ",
			*(size_t *)aml_layout_deref(layout_elements,
						    (size_t[]){1, i, 0}));
		fprintf(stderr, "%ld ",
			*(size_t *)aml_layout_deref(layout_elements,
						    (size_t[]){2, i, 0}));
		fprintf(stderr, "%lf ",
			*(double *)aml_layout_deref(layout_elements,
						    (size_t[]){3, i, 0}));
	}
	fprintf(stderr, "\n");

	fprintf(stderr, "Changing the shape of the layout...\n");
	size_t *array_coords;

	// We get a new memory allocation for this new layout
	array_coords = malloc(sizeof(struct particle) * size);

	assert(!aml_layout_dense_create(&new_layout, array_coords,
					AML_LAYOUT_ORDER_COLUMN_MAJOR, //FORTRAN
					sizeof(size_t), 3,
					(size_t[]){size_1, size_2, 4},
					NULL, NULL));

	assert(!aml_dma_linux_transform_generic(new_layout,
		layout_elements, NULL));

	/* Let's check we now have a struct of arrays by looking at the first
	 * elements */
	for (size_t i = 0; i < 10; i++)
		fprintf(stderr, "%ld ",
			*(size_t *)aml_layout_deref(new_layout,
						    (size_t[]){i, 0, 0}));
	fprintf(stderr, "\n");

	/* Getting only the energy of the particles */
	size_t offsets[3] = {0, 0, 3};
	struct aml_layout *layout_energy;

	fprintf(stderr, "Looking only at the energy now...\n");

	/* This is done by slicing the layout, keeping only the dimensions we
	 * want */
	assert(!aml_layout_slice(new_layout, &layout_energy,
				 offsets, (size_t[]){size_1, size_2, 1},
				 NULL));

	for (size_t i = 0; i < 10; i++)
		fprintf(stderr, "%lf ",
			*(double *)aml_layout_deref(layout_energy,
						    (size_t[]){i, 0, 0}));
	fprintf(stderr, "\n");


	aml_layout_dense_destroy(&lay_part);
	aml_layout_dense_destroy(&layout_elements);
	aml_layout_dense_destroy(&new_layout);
	aml_layout_dense_destroy(&layout_energy);
	aml_area_munmap(area, particles, sizeof(struct particle) * size);
	free(array_coords);

	aml_finalize();

	return 0;
}
