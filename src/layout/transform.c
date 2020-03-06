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
#include "aml/layout/transform.h"
#include "aml/layout/native.h"

//------------------------------------------------------------------------------
// Coordinates Utils
//------------------------------------------------------------------------------

void
increment_coords(const size_t ndims,
		 const size_t *dims, size_t *coords, size_t n)
{
	for (size_t c, r, j = 0; j < ndims && n > 0; j++) {
		c = coords[j];
		r = (c + n) % dims[j];
		n = (c + n) / dims[j];
		coords[j] = r;
	}
}

void
decrement_coords(const size_t ndims,
		 const size_t *dims, size_t *coords, size_t n)
{
	for (size_t c, r, j = 0; j < ndims && n > 0; j++) {
		c = coords[j];
		if (n > c) {
			r = dims[j] - (n + c) % dims[j];
			n = 1 + n / dims[j];
			coords[j] = r;
		} else {
			n = 0;
			coords[j] = c - n;
		}
	}
}

void index_to_coords(size_t n, size_t *coords,
		     const size_t *dims, const size_t ndims)
{
	for (size_t i = 0; i < ndims; i++) {
		coords[i] = n % dims[i];
		n -= coords[i];
		n /= dims[i];
	}
}

//------------------------------------------------------------------------------
// Generic Layout Iterator
//------------------------------------------------------------------------------

/** Iterator on elements of two layouts **/
struct aml_layout_transform_iterator {
	/** Layout destination of the transform **/
	const struct aml_layout *src;
	/** Layout destination of the transform **/
	struct aml_layout *dst;
	/** Size of an element in layouts **/
	size_t element_size;
	/** The number of elements in the iterator **/
	size_t num_elements;
	/** The index of current element **/
	size_t cur_element;
	/** src num dimensions **/
	size_t src_ndims;
	/** dst num dimensions **/
	size_t dst_ndims;
	/** src dimensions **/
	size_t *src_dims;
	/** dst dimensions **/
	size_t *dst_dims;
};

/** Item returned on transform iterator steps. **/
struct aml_layout_transform_item {
	/** Pointer to contiguous memory in src layout **/
	void *src;
	/** Pointer to contiguous memory in dst layout **/
	void *dst;
	/** Size iterated by item **/
	size_t size;
};

#define ITERATOR_TYPES		\
	struct aml_layout *,	\
		struct aml_layout *,\
		size_t,				\
		size_t,				\
		size_t,				\
		size_t,				\
		size_t,				\
		size_t *,			\
		size_t *

/**
 * Iterator constructor.
 * @param out[out]: Pointer to iterator to alloc.
 * @param dst[in]: The destination layout of transform.
 * @param src[in]: The destination layout of transform.
 * src and dst must have the same number of elements and same element sizes.
 **/
static int
aml_layout_transform_iterator_create(struct aml_layout_transform_iterator **out,
				     struct aml_layout *dst,
				     const struct aml_layout *src)
{
	unsigned long nelem_src = 1, nelem_dst = 1;
	size_t esize;
	size_t ndims_src, ndims_dst;
	struct aml_layout_transform_iterator *ret;
	int err = AML_SUCCESS;

	if (out == NULL || dst == NULL || src == NULL)
		return -AML_EINVAL;

	esize = aml_layout_element_size(src);
	if (aml_layout_element_size(src) != esize)
		return -AML_EINVAL;

	ndims_src = aml_layout_ndims(src);
	ndims_dst = aml_layout_ndims(dst);

	ret =
	    AML_INNER_MALLOC_ARRAY((ndims_src + ndims_dst), size_t,
				   ITERATOR_TYPES);

	if (ret == NULL)
		return -AML_ENOMEM;

	ret->src = src;
	ret->dst = dst;
	ret->element_size = esize;
	ret->src_ndims = ndims_src;
	ret->dst_ndims = ndims_dst;

	ret->src_dims = AML_INNER_MALLOC_GET_ARRAY(ret, size_t, ITERATOR_TYPES);
	ret->dst_dims = &(ret->src_dims[ndims_src]);

	err = aml_layout_dims_native(src, ret->src_dims);
	if (err != AML_SUCCESS)
		goto failure;
	err = aml_layout_dims_native(dst, ret->dst_dims);
	if (err != AML_SUCCESS)
		goto failure;

	for (size_t i = 0; i < ndims_src; i++)
		nelem_src *= ret->src_dims[i];
	for (size_t i = 0; i < ndims_dst; i++)
		nelem_dst *= ret->dst_dims[i];
	if (nelem_dst != nelem_src) {
		err = -AML_EINVAL;
		goto failure;
	}
	ret->num_elements = nelem_dst;
	ret->cur_element = 0;

	*out = ret;
	return AML_SUCCESS;
failure:
	free(ret);
	return err;
}

/** Iterator destructor. *it is set to NULL. **/
static int
aml_layout_transform_iterator_destroy(struct aml_layout_transform_iterator **it)
{
	if (it == NULL || *it == NULL)
		return -AML_EINVAL;
	free(*it);
	*it = NULL;
	return AML_SUCCESS;
}

/**
 * Store current element pointed by iterator in item.
 * @param it[in]: A non NULL iterator.
 * @param item[out]: An item where to store resulting pointers.
 * If NULL, nothing is stored.
 * @return -AML_EINVAL if "it" is NULL.
 * @return -AML_EDOM if iterator is beyond last element.
 * @return AML_SUCCESS otherwise.
 **/
static int
aml_layout_transform_iterator_current(const struct aml_layout_transform_iterator
				      *it,
				      struct aml_layout_transform_item *item)
{
	if (it == NULL)
		return -AML_EINVAL;

	if (it->cur_element >= it->num_elements)
		return -AML_EDOM;

	if (item != NULL) {
		size_t csrc[it->src_ndims];
		size_t cdst[it->dst_ndims];

		index_to_coords(it->cur_element, csrc, it->src_dims,
				it->src_ndims);
		index_to_coords(it->cur_element, cdst, it->dst_dims,
				it->dst_ndims);
		item->src = aml_layout_deref_native(it->src, csrc);
		item->dst = aml_layout_deref_native(it->dst, cdst);
		item->size = it->element_size;
	}
	return AML_SUCCESS;
}

/**
 * Reset iterator and store the first element in item
 * @code
 * struct aml_layout_transform_item item;
 * for (int err = aml_layout_transform_iterator_begin(it, &item);
 *      err == AML_SUCCESS;
 *      err = aml_layout_transform_iterator_step(it, n, &item)) { ... }
 * @endcode
 **/
static int
aml_layout_transform_iterator_begin(struct aml_layout_transform_iterator *it,
				    struct aml_layout_transform_item *item)
{
	if (it == NULL)
		return -AML_EINVAL;

	it->cur_element = 0;
	return aml_layout_transform_iterator_current(it, item);
}

/**
 * Step an arbitrary amount of steps, forward or backward.
 * @param it[in]: The iterator on which we iterate.
 * @param n[in]: The amount of steps to move. If negative, move backward.
 * @param next[out]: The item where to store iteration result.
 * If NULL, nothing is stored.
 * @return -AML_EINVAL if "it" is NULL.
 * @return -AML_EDOM if iterator goes is beyond last or first element.
 * @return AML_SUCCESS otherwise.
 **/
static int
aml_layout_transform_iterator_step(struct aml_layout_transform_iterator *it,
				   const ssize_t n,
				   struct aml_layout_transform_item *next)
{
	if (it == NULL)
		return -AML_EINVAL;

	if (n > 0 && (size_t) (n + it->cur_element) > it->num_elements)
		return -AML_EDOM;

	if (n < 0 && (size_t) (-1 * n) > it->cur_element)
		return -AML_EDOM;

	it->cur_element = it->cur_element + n;
	return aml_layout_transform_iterator_current(it, next);
}

/**
 * Move iterator to next non-contiguous element.
 * @param it[in]: The iterator on which we iterate.
 * @param next[out]: The item where to store iteration result.
 * @return -AML_EINVAL if "it" is NULL.
 * @return -AML_EDOM if iterator goes is beyond last or first element.
 * @return AML_SUCCESS otherwise.
 **/
static int
aml_layout_transform_iterator_next_contiguous(
	struct aml_layout_transform_iterator *it,
	struct aml_layout_transform_item *next)
{

	if (it == NULL || next == NULL)
		return -AML_EINVAL;

	int err = aml_layout_transform_iterator_current(it, next);

	if (err != AML_SUCCESS)
		return err;

	struct aml_layout_transform_item item = { next->src,
		next->dst,
		next->size
	};

	for (err = aml_layout_transform_iterator_step(it, 1, &item);
	     err == AML_SUCCESS;
	     err = aml_layout_transform_iterator_step(it, 1, &item)) {
		if (((size_t) item.src - (size_t) next->src) != next->size ||
		    ((size_t) item.dst - (size_t) next->dst) != next->size)
			break;
		next->size += item.size;
	}

	return AML_SUCCESS;
}

//------------------------------------------------------------------------------
// Generic Transform
//------------------------------------------------------------------------------

int
aml_layout_transform(struct aml_layout *dst,
					 const struct aml_layout *src,
					 struct aml_transform_args *args)
{
	struct aml_layout_transform_iterator *it;
	struct aml_layout_transform_item item;

	aml_layout_transform_iterator_create(&it, dst, src);
	aml_layout_transform_iterator_begin(it, &item);

	for (int err = aml_layout_transform_iterator_next_contiguous(it, &item);
	     err == AML_SUCCESS;
	     err = aml_layout_transform_iterator_next_contiguous(it, &item)) {
		if (args->memcpy(item.dst, item.src, item.size,
						 args->data) != AML_SUCCESS)
			return -AML_FAILURE;
	}

	aml_layout_transform_iterator_destroy(&it);
	return AML_SUCCESS;
}

int
aml_layout_copy(struct aml_layout *dst,
				const struct aml_layout *src,
				struct aml_transform_args *args)
{
	int err;
	void *dst_ptr = aml_layout_rawptr(dst);
	void *src_ptr = aml_layout_rawptr(src);
	size_t src_ndims = aml_layout_ndims(src);
	size_t dst_ndims = aml_layout_ndims(dst);
	size_t src_dims[src_ndims];
	size_t dst_dims[dst_ndims];
	size_t src_size = aml_layout_element_size(src);
	size_t dst_size = aml_layout_element_size(dst);

	err = aml_layout_dims(dst, src_dims);
	if (err != AML_SUCCESS)
		return err;
	err = aml_layout_dims(dst, dst_dims);
	if (err != AML_SUCCESS)
		return err;
	for (size_t i = 0; i < src_ndims; i++)
		src_size *= src_dims[i];
	for (size_t i = 0; i < dst_ndims; i++)
		dst_size *= dst_dims[i];

	if (src_size != dst_size)
		return -AML_EINVAL;

	args->memcpy(dst_ptr, src_ptr, src_size, args->data);
	return AML_SUCCESS;
}

//------------------------------------------------------------------------------
// Memcpy functors implementation
//------------------------------------------------------------------------------

// Linux memcpy
int aml_linux_memcpy(void *dst, const void *src, size_t size, void *args)
{
	(void) args;
	memcpy(dst, src, size);
	return AML_SUCCESS;
}

