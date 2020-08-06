/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

/**
 * This file contains helper functions on layouts.
 **/

#include "test_layout.h"
#include "aml/layout/native.h"

int aml_layout_fill_with_coords(struct aml_layout *layout)
{
	int err;
	size_t ndims = aml_layout_ndims(layout);
	size_t dims[ndims];
	size_t coords[ndims];
	size_t bits[ndims];
	size_t nelems = 1;

	assert(aml_layout_element_size(layout) >= 8);
	err = aml_layout_dims(layout, dims);
	if (err != AML_SUCCESS)
		return err;

	// count number of element.
	for (size_t i = 0; i < ndims; i++) {
		coords[i] = 0;
		nelems *= dims[i];
	}

	// Packing coordinates space.
	dims_nbits(ndims, dims, bits);

	for (size_t i = 0; i < nelems; i++) {
		// Get element
		uint64_t *e = (uint64_t *) aml_layout_deref(layout, coords);
		// Pack and store coordinates.
		*e = pack_coords(ndims, coords, bits);
		// Move to next coordinates.
		increment_coords(ndims, dims, coords, 1);
	}

	return AML_SUCCESS;
}

int aml_layout_isequal(const struct aml_layout *a,
		       const struct aml_layout *b)
{
	// Check order equality
	if (aml_layout_order(a) != aml_layout_order(b))
		return 0;

	// Check number of dimensions equality
	size_t ndims_a = aml_layout_ndims(a);
	size_t ndims_b = aml_layout_ndims(b);

	if (ndims_a != ndims_b)
		return 0;

	// Check dimensions equality
	size_t dims_a[ndims_a];
	size_t dims_b[ndims_b];
	size_t nelem = 1;

	assert(aml_layout_dims(a, dims_a) == AML_SUCCESS);
	assert(aml_layout_dims(b, dims_b) == AML_SUCCESS);
	for (size_t i = 0; i < ndims_a; i++) {
		if (dims_a[i] != dims_b[i])
			return 0;
		nelem *= dims_a[i];
	}

	// Check elements equality
	size_t coords[ndims_a];

	for (size_t i = 0; i < ndims_a; i++)
		coords[i] = 0;

	for (size_t i = 0; i < nelem; i++) {
		if (aml_layout_deref(a, coords) != aml_layout_deref(b, coords))
			return 0;
		increment_coords(ndims_a, dims_a, coords, 1);
	}

	return 1;
}

void test_layout_base(struct aml_layout *layout)
{
	int order = aml_layout_order(layout);

	assert(order == AML_LAYOUT_ORDER_COLUMN_MAJOR ||
	       order == AML_LAYOUT_ORDER_ROW_MAJOR);

	size_t ndims = aml_layout_ndims(layout);
	size_t dims[ndims];
	size_t dims_native[ndims];

	assert(aml_layout_dims_native(layout, dims_native) == AML_SUCCESS);
	assert(aml_layout_dims(layout, dims) == AML_SUCCESS);

	size_t element_size = aml_layout_element_size(layout);

	assert(element_size != 0);

	size_t n_elements = 1;
	size_t coords[ndims];
	size_t coords_native[ndims];

	for (size_t i = 0; i < ndims; i++) {
		n_elements *= dims[i];
		coords[i] = 0;
		coords_native[i] = 0;
	}

	// Try deref/deref_native on all elements
	for (size_t i = 0; i < n_elements; i++) {
		assert(aml_layout_deref_native(layout, coords_native) != NULL);
		increment_coords(ndims, dims_native, coords_native, 1);
		assert(aml_layout_deref(layout, coords) != NULL);
		increment_coords(ndims, dims, coords, 1);
	}
}

void test_layout_fprintf(FILE *stream, const char *prefix,
			 struct aml_layout *layout)
{
	assert(!aml_layout_fprintf(stream, prefix, layout));
}

void test_layout_duplicate(struct aml_layout *layout)
{
	struct aml_layout *l;
	assert(!aml_layout_duplicate(layout, &l));
	aml_layout_destroy(&l);
}
