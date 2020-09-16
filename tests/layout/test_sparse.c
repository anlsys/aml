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

#include "aml/layout/sparse.h"

void test_common(struct aml_layout *layout, size_t nptr, size_t *data)
{
	size_t dims;

	assert(aml_layout_order(layout) == AML_LAYOUT_ORDER_ROW_MAJOR);
	assert(aml_layout_ndims(layout) == 1);
	assert(aml_layout_element_size(layout) == sizeof(void *));
	assert(aml_layout_dims(layout, &dims) == AML_SUCCESS);
	assert(dims == nptr);
	for (size_t coords = 0; coords < nptr; coords++)
		assert(aml_layout_deref(layout, &coords) == data + coords);
}

void test_functional()
{
	// Data initialization
	const size_t nptr = 16;
	size_t data[nptr];
	void *ptrs[nptr];
	size_t sizes[nptr];

	for (size_t i = 0; i < nptr; i++) {
		data[i] = i;
		sizes[i] = sizeof(*sizes);
		ptrs[i] = data + i;
	}

	// Create layout
	struct aml_layout *layout;
	assert(aml_layout_sparse_create(&layout, nptr, ptrs, sizes, NULL, 0) ==
	       AML_SUCCESS);
	test_common(layout, nptr, data);

	// Duplicate
	struct aml_layout *dup;
	assert(aml_layout_duplicate(layout, &dup, NULL) == AML_SUCCESS);
	test_common(dup, nptr, data);

	// Cleanup
	aml_layout_destroy(&layout);
	aml_layout_destroy(&dup);
}

int main()
{
	test_functional();
	return 0;
}
