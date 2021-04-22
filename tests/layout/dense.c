/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

/**
 * This file defines the functions to test on a supposedly dense layout.
 **/

#include "test_layout.h"

/**
 * Slice an hyperplan of a layout along one dimension.
 * @param layout: the layout to slice.
 * @param dim: the dimension out of hyperplan.
 * @param hindex: The index in dim for selecting an hyperplan.
 **/
static struct aml_layout *slice_hyperplan(struct aml_layout *layout,
					  size_t dim, size_t hindex)
{
	struct aml_layout *slice;
	size_t ndims = aml_layout_ndims(layout);
	size_t dims[ndims];
	size_t new_dims[ndims];
	size_t offsets[ndims];

	// Compute dimensions. They are the same except on hyperplan dim
	// where it is 1.
	assert(aml_layout_dims(layout, dims) == AML_SUCCESS);
	for (size_t i = 0; i < ndims; i++)
		new_dims[i] = dims[i];
	new_dims[dim] = 1;

	// On hyper plan dimension the slice is offseted by the hyperplan index
	for (size_t i = 0; i < ndims; i++)
		offsets[i] = 0;
	offsets[dim] = hindex;

	assert(!aml_layout_slice(layout, &slice, offsets, new_dims, NULL));

	size_t dims_slice[ndims];

	// Check slice dimensions are the same as the one requested.
	assert(aml_layout_dims(slice, dims_slice) == AML_SUCCESS);
	assert(!memcmp(new_dims, dims_slice, sizeof(dims_slice)));

	return slice;
}

/**
 * Test that all elements of an hyperplan have the same value as
 * hyperplan index.
 * @param layout: the layout to slice.
 * @param dim: the dimension out of hyperplan.
 * @param hindex: The index in dim of the hyperplan.
 **/
static void test_slice_hyperplan(struct aml_layout *layout, size_t dim,
				 size_t hindex)
{
	struct aml_layout *slice = slice_hyperplan(layout, dim, hindex);

	assert(slice != NULL);
	assert(aml_layout_order(layout) == aml_layout_order(slice));

	size_t size = 1;
	size_t ndims = aml_layout_ndims(layout);
	size_t dims[ndims];
	size_t dims_slice[ndims];
	size_t coords[ndims];
	size_t coords_slice[ndims];
	size_t bits[ndims];

	assert(aml_layout_dims(layout, dims) == AML_SUCCESS);
	assert(aml_layout_dims(slice, dims_slice) == AML_SUCCESS);
	dims_nbits(ndims, dims, bits);
	for (size_t i = 0; i < ndims; i++) {
		coords[i] = 0;
		coords_slice[i] = 0;
		size *= dims[i];
	}

	// For each element of the slice, check that coordinates stored
	// inside value, match with the hyperplan coordinates.
	for (size_t i = 0; i < size; i++) {
		uint64_t *e = aml_layout_deref(slice, coords_slice);

		unpack_coords(*e, ndims, bits, coords);
		assert(coords[dim] == hindex);
		increment_coords(ndims, dims_slice, coords_slice, 1);
	}

	free(slice);
}

void test_slice_dense(struct aml_layout *layout)
{
	assert(aml_layout_fill_with_coords(layout) == AML_SUCCESS);

	size_t ndims = aml_layout_ndims(layout);
	size_t dims[ndims];

	assert(aml_layout_dims(layout, dims) == AML_SUCCESS);

	// Define the hyperplans case to check.
	// We don't check all possible hyperplans to make it faster.
	// We check first, middle and last dimensions as hyperplan dimensions.
	// In each dimension, we check first, middle and last element of the
	// dimension as hyperplan index.
	size_t cases[9][2] = {
		{0, 0}, {0, dims[0] / 2}, {0, dims[0] - 1},
		{ndims / 2, 0}, {ndims / 2, dims[ndims / 2] / 2}, {ndims / 2,
								   dims[ndims /
									2] - 1},
		{ndims - 1, 0}, {ndims - 1, dims[ndims - 1] / 2}, {ndims - 1,
								   dims[ndims -
									1] - 1}
	};

	for (size_t c = 0; c < 9; c++)
		test_slice_hyperplan(layout, cases[c][0], cases[c][1]);
}
