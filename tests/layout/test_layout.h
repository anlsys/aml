/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#ifndef AML_LAYOUT_TEST_H
#define AML_LAYOUT_TEST_H

#include "aml.h"
#include "aml/layout/dense.h"

//------------------------------------------------------------------------------
// Packing/Unpacking/Incrementing layout coordinates
//------------------------------------------------------------------------------

/**
 * Count the amount of bits required to store each dimension
 * coordinate.
 * @param ndims[in]: The number of dimensions.
 * @param dims[in]: The number of element in each dimensions.
 * @param bits[out]: An array counting the number of bits required
 * to store each dimension coordinate.
 **/
void dims_nbits(const size_t ndims,
		const size_t *dims,
		size_t *bits);

/**
 * Take an array of coordinates and the storage size they require and
 * store coordinates in a uint64_t.
 * @param ndims[in]: The number of coordinates.
 * @param coords[in]: The coordinates to pack.
 * @param bits[in]: The number of bits required to store each coordinates.
 * The sum of bits must be less or equal to 64.
 * @return The packed coordinates.
 **/
uint64_t pack_coords(const size_t ndims,
		     const size_t *coords,
		     const size_t *bits);
/**
 * Take a set of coordinates packed into a uint64_t and
 * return an array of coodinates.
 * @param coords[in]: The set of coordinates packed in a uint64_t.
 * @param ndims[in]: The number of coordinates.
 * @param bits[in]: The number of bits required to store each coordinates.
 * The sum of bits must be less or equal to 64.
 * @param out[out]: For each dimension, the coordinate in the dimension.
 **/
void unpack_coords(uint64_t coords,
		   const size_t ndims,
		   const size_t *bits,
		   size_t *out);

/**
 * Increment a set of coordinates by n, starting from last dimension.
 * @param ndims[in]: The number of coordinates.
 * @param dims[in]: The maximum coordinate in each dimension.
 * @param coords[in/out]: The coordinates to increment in place.
 * @param n[in]: The increment size. In case of overflow, the increment
 * will cycle.
 **/
void increment_coords(const size_t ndims,
		      const size_t *dims,
		      size_t *coords,
		      size_t n);

//------------------------------------------------------------------------------
// General layout helper functions.
//------------------------------------------------------------------------------

/**
 * Set a layout elements with their own packed coordinates.
 * Layout element size must be greater or equal to `sizeof(uint64_t)`.
 * @param layout[in/out]: The layout to fill.
 **/
int aml_layout_fill_with_coords(struct aml_layout *layout);

/**
 * Check if:
 * - order are equal
 * - ndims are equal
 * - dims are equal
 * - deref all elements give same pointers.
 * @param a[in]: Left hand side to check.
 * @param b[in]: Right hand side to check.
 **/
int aml_layout_isequal(const struct aml_layout *a,
		       const struct aml_layout *b);
/**
 * Check if:
 * - order are equal
 * - ndims are equal
 * - dims are equal
 * - deref all elements have the same value.
 * @param a[in]: Left hand side to check.
 * @param b[in]: Right hand side to check.
 **/
int
aml_layout_isequivalent(const struct aml_layout *a, const struct aml_layout *b);

/**
 * Base test for layouts. Check that most API methods work.
 * Slice and reshape are not checked.
 **/
void test_layout_base(struct aml_layout *layout);

/**
 * Create a layout dense that copy dimensions of a layout in a dense layout.
 * aml_layout_isequal(in, out) returns true.
 * @param in[in]: The layout to copy.
 * @param out[out]: The dense copy.
 * @param ptr[out]: The pointer inside layout to free.
 **/
int
test_layout_dense_copy(const struct aml_layout *in, struct aml_layout **out);

//------------------------------------------------------------------------------
// Testing a dense layout.
//------------------------------------------------------------------------------

/**
 * Slices a dense layout into hyperplans. Then, check
 * that the sublayout dereferences elements in the good dimension.
 * !Use only on dense layout.
 **/
void test_slice_dense(struct aml_layout *layout);

//------------------------------------------------------------------------------
// Initializing some test layouts
//------------------------------------------------------------------------------

/** Declare and implement a function for initializing a dense layout. **/
#define LAYOUT_DENSE_CREATE_DECL(name, ndims, ...)\
static struct aml_layout *name()\
{\
	struct aml_layout *l = NULL;\
	STATIC_ARRAY_DECL(size_t, dims, ndims, __VA_ARGS__, 0);\
	STATIC_ARRAY_DECL(size_t, stride, ndims, SKIP(ndims, __VA_ARGS__, 0));\
	size_t n_elems = 1;\
	uint64_t *memory;\
	size_t elem_size = sizeof(*memory);\
	for (size_t i = 0; i < ndims; i++) \
		n_elems *= (stride[i] * dims[i]); \
	memory = malloc(elem_size * n_elems);\
	if (memory == NULL)\
		return NULL;\
	if (aml_layout_dense_create(&l,\
		(void *)memory, AML_LAYOUT_ORDER_COLUMN_MAJOR,\
		sizeof(*memory), ndims, dims,\
		stride, NULL) != AML_SUCCESS) { \
		free(memory);\
		return NULL;\
	} \
	return l;\
}

/** Free a dense layout and its inner data. **/
void layout_dense_free(struct aml_layout *l);

//------------------------------------------------------------------------------
// Testing a layout reshape method against struct aml_layout_reshape.
// Also testing struct aml_layout_reshape on its own.
//------------------------------------------------------------------------------

/**
 * Reshape a layout with abstraction method and with
 * `struct aml_layout_reshape` and check their equality.
 **/
void test_layout_reshape(struct aml_layout *layout,
			 const size_t n_new_dims,
			 const size_t *new_dims);

#endif //AML_LAYOUT_TEST_H
