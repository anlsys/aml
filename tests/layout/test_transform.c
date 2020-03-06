/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include <assert.h>
#include "aml.h"
#include "aml/layout/dense.h"
#include "aml/layout/native.h"
#include "aml/dma/linux-seq.h"
#include "test_layout.h"

/**
 * Walk all elements one by one and check that the transform gave the
 * same elements.
 * t(a,b) -> a
 */
void
test_base(struct aml_layout *a, struct aml_layout *b)
{
	size_t size    = 1;
	size_t esize   = aml_layout_element_size(a);
	size_t a_ndims = aml_layout_ndims(a);
	size_t b_ndims = aml_layout_ndims(b);
	size_t a_dims[a_ndims];
	size_t b_dims[b_ndims];
	size_t a_coords[a_ndims];
	size_t b_coords[b_ndims];
	void *a_ptr, *b_ptr;

	// Initialization
	assert(aml_layout_dims(a, a_dims) == AML_SUCCESS);
	assert(aml_layout_dims(b, b_dims) == AML_SUCCESS);
	assert(aml_layout_fill_with_coords(a) == AML_SUCCESS);
	assert(aml_layout_fill_with_coords(b) == AML_SUCCESS);
	for (size_t i = 0; i < a_ndims; i++)
		size *= a_dims[i];
	for (size_t i = 0; i < a_ndims; i++)
		a_coords[i] = 0;
	for (size_t i = 0; i < b_ndims; i++)
		b_coords[i] = 0;

	// Transform
	aml_dma_linux_transform_generic(b, a, NULL);

	// Test
	for (size_t i = 0; i < size; i++) {
		a_ptr = aml_layout_deref(a, a_coords);
		b_ptr = aml_layout_deref(b, b_coords);
	    assert(!memcmp(a_ptr, b_ptr, esize));
		increment_coords(a_ndims, a_dims, a_coords, 1);
		increment_coords(b_ndims, b_dims, b_coords, 1);
	}
}

/**
 * transform "a" into "b", then "b" back into "a". Check that values in "a"
 * reflect the original values in "a".
 * t(a, t(b,a)) -> a
 **/
void
test_symmetry(struct aml_layout *a, struct aml_layout *b)
{

	size_t size  = 1;
	size_t ndims = aml_layout_ndims(a);
	size_t dims[ndims];
	size_t coords[ndims];
	uint64_t packed_coords, unpacked_coords;
	size_t bits[ndims];

	// Fill a with element coordinates.
	assert(aml_layout_dims(a, dims) == AML_SUCCESS);
	dims_nbits(ndims, dims, bits);
	assert(aml_layout_fill_with_coords(a) == AML_SUCCESS);

	// Transform a into b.
	aml_dma_linux_transform_generic(b, a, NULL);
	// Then b back into a.
	aml_dma_linux_transform_generic(a, b, NULL);

	for (size_t i = 0; i < ndims; i++)
		coords[i] = 0;
	for (size_t i = 0; i < ndims; i++)
		size *= dims[i];

	// Check that alement into a match the original ones.
	// True <=> elements values hold their packed coordinates.
	for (size_t i = 0; i < size; i++) {
		packed_coords   = *(uint64_t *) aml_layout_deref(a, coords);
		unpacked_coords = pack_coords(ndims, coords, bits);
		assert(packed_coords == unpacked_coords);
		increment_coords(ndims, dims, coords, 1);
	}
}

/**
 * Check that transforming is transitive, i.e "a" to "b" to "c" is
 * equivalent to transforming "a" to "c".
 * t(t(a,b),c) <-> t(a,c)
 **/
void
test_transitivity(struct aml_layout *a,
		  struct aml_layout *b,
		  struct aml_layout *c)
{
	struct aml_layout *c1;

	// b <- a
	aml_dma_linux_transform_generic(b, a, NULL);
	// c <- b (aka a)
	aml_dma_linux_transform_generic(c, b, NULL);
	// Store a copy of c in c1.
	test_layout_dense_copy(c, &c1);
	// c <- a
	aml_dma_linux_transform_generic(c, a, NULL);
	// c and c1 should be identical
	assert(aml_layout_isequivalent(c, c1));

	layout_dense_free(c1);
}

void
test_transform_all(struct aml_layout *a,
		   struct aml_layout *b,
		   struct aml_layout *c)
{
	test_base(a, b);
	test_symmetry(a, b);
	test_transitivity(a, b, c);
}

LAYOUT_DENSE_CREATE_DECL(make_layout_dense_stride,
5, 2, 3, 6, 10, 13, 1, 2, 1, 1, 1)

LAYOUT_DENSE_CREATE_DECL(make_layout_dense,
6, 2, 3, 2, 3, 10, 13, 1, 1, 1, 1, 1, 1)

LAYOUT_DENSE_CREATE_DECL(make_layout_dense_stride_2,
3, 6, 12, 65, 3, 1, 1)

int
main()
{
	// Test for dense layout
	struct aml_layout *a = make_layout_dense_stride();
	struct aml_layout *b = make_layout_dense();
	struct aml_layout *c = make_layout_dense_stride_2();

	test_transform_all(a, b, c);

	layout_dense_free(a);
	layout_dense_free(b);
	layout_dense_free(c);
	return 0;
}
