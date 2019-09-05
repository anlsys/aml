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
#include "aml/layout/native.h"
#include "aml/tiling/resize.h"

/*******************************************************************************
 * Create/Destroy
 ******************************************************************************/

static int aml_tiling_resize_alloc(struct aml_tiling **ret, size_t ndims)
{
	struct aml_tiling *tiling;
	struct aml_tiling_resize *data;

	tiling = AML_INNER_MALLOC_EXTRA(struct aml_tiling,
					struct aml_tiling_resize,
					size_t, 3*ndims);

	if (tiling == NULL) {
		*ret = NULL;
		return -AML_ENOMEM;
	}

	data = AML_INNER_MALLOC_NEXTPTR(tiling,
					struct aml_tiling,
					struct aml_tiling_resize);
	tiling->data = (struct aml_tiling_data *)data;
	data->tile_dims = AML_INNER_MALLOC_EXTRA_NEXTPTR(tiling,
						 struct aml_tiling,
						 struct aml_tiling_resize,
						 size_t, 0);
	data->dims = AML_INNER_MALLOC_EXTRA_NEXTPTR(tiling,
						    struct aml_tiling,
						    struct aml_tiling_resize,
						    size_t, ndims);
	data->border_tile_dims = AML_INNER_MALLOC_EXTRA_NEXTPTR(tiling,
						    struct aml_tiling,
						    struct aml_tiling_resize,
						    size_t, 2*ndims);
	data->layout = NULL;
	data->ndims = ndims;
	*ret = tiling;
	return 0;
}

int aml_tiling_resize_create(struct aml_tiling **tiling,
			     const int tags,
			     const struct aml_layout *layout,
			     size_t ndims, const size_t *tile_dims)
{
	struct aml_tiling *t;
	struct aml_tiling_resize *data;
	int err;

	if (tiling == NULL || layout == NULL || tile_dims == NULL || !ndims)
		return -AML_EINVAL;

	err = aml_tiling_resize_alloc(&t, ndims);
	if (err)
		return err;

	data = (struct aml_tiling_resize *)t->data;
	data->layout = layout;

	switch (AML_TILING_ORDER(tags)) {

	case AML_TILING_ORDER_ROW_MAJOR:
		t->ops = &aml_tiling_resize_row_ops;
		for (size_t i = 0; i < ndims; i++)
			data->tile_dims[i] = tile_dims[ndims-i-1];
		break;
	case AML_TILING_ORDER_COLUMN_MAJOR:
		t->ops = &aml_tiling_resize_column_ops;
		for (size_t i = 0; i < ndims; i++)
			data->tile_dims[i] = tile_dims[i];
		break;
	default:
		free(t);
		*tiling = NULL;
		return -AML_EINVAL;
	}
	*tiling = t;
	data->tags = tags;
	size_t target_dims[ndims];

	aml_layout_dims_native(layout, target_dims);
	for (size_t i = 0; i < ndims; i++) {
		data->border_tile_dims[i] = target_dims[i] % data->tile_dims[i];
		data->dims[i] = target_dims[i] / data->tile_dims[i];
		if (data->border_tile_dims[i] == 0)
			data->border_tile_dims[i] = data->tile_dims[i];
		else
			data->dims[i] += 1;
	}
	return 0;
}

/*******************************************************************************
 * Column Implementation
 ******************************************************************************/
struct aml_layout*
aml_tiling_resize_column_index(const struct aml_tiling_data *t,
			       const size_t *coords)
{
	const struct aml_tiling_resize *d =
	    (const struct aml_tiling_resize *)t;
	struct aml_layout *ret;

	assert(d != NULL);
	size_t ndims = d->ndims;
	size_t offsets[ndims];
	size_t dims[ndims];
	size_t strides[ndims];

	for (size_t i = 0; i < ndims; i++)
		assert(coords[i] < d->dims[i]);
	for (size_t i = 0; i < ndims; i++) {
		offsets[i] = coords[i] * d->tile_dims[i];
		strides[i] = 1;
	}
	for (size_t i = 0; i < ndims; i++)
		dims[i] = (coords[i] == d->dims[i] - 1 ?
			      d->border_tile_dims[i] :
			      d->tile_dims[i]);

	aml_layout_slice_native(d->layout, &ret, offsets, dims, strides);
	return ret;
}

struct aml_layout*
aml_tiling_resize_column_index_linear(const struct aml_tiling_data *t,
				      const size_t uuid)
{
	(void)t;
	(void)uuid;
	return NULL;
}

int aml_tiling_resize_column_order(const struct aml_tiling_data *t)
{
	(void)t;
	return AML_TILING_ORDER_COLUMN_MAJOR;
}

int aml_tiling_resize_column_tile_dims(const struct aml_tiling_data *t,
					size_t *tile_dims)
{
	const struct aml_tiling_resize *d =
	    (const struct aml_tiling_resize *)t;
	assert(d != NULL);
	memcpy((void *)tile_dims, (void *)d->tile_dims,
	       sizeof(size_t)*d->ndims);
	return 0;
}

int aml_tiling_resize_column_dims(const struct aml_tiling_data *l,
				  size_t *dims)
{
	const struct aml_tiling_resize *d =
	    (const struct aml_tiling_resize *)l;
	assert(d != NULL);
	memcpy((void *)dims, (void *)d->dims, sizeof(size_t)*d->ndims);
	return 0;
}

size_t aml_tiling_resize_column_ndims(const struct aml_tiling_data *l)
{
	const struct aml_tiling_resize *d =
	    (const struct aml_tiling_resize *)l;
	assert(d != NULL);
	return d->ndims;
}

size_t aml_tiling_resize_column_ntiles(const struct aml_tiling_data *l)
{
	const struct aml_tiling_resize *d =
	    (const struct aml_tiling_resize *)l;
	assert(d != NULL);
	return 0;
}

struct aml_tiling_ops aml_tiling_resize_column_ops = {
	aml_tiling_resize_column_index,
	aml_tiling_resize_column_index_linear,
	aml_tiling_resize_column_order,
	aml_tiling_resize_column_tile_dims,
	aml_tiling_resize_column_dims,
	aml_tiling_resize_column_ndims,
	aml_tiling_resize_column_ntiles,
};

/*******************************************************************************
 * Column Implementation
 ******************************************************************************/

struct aml_layout*
aml_tiling_resize_row_index(const struct aml_tiling_data *t,
			    const size_t *coords)
{
	const struct aml_tiling_resize *d =
	    (const struct aml_tiling_resize *)t;
	struct aml_layout *ret;

	assert(d != NULL);

	size_t ndims = d->ndims;
	size_t offsets[ndims];
	size_t dims[ndims];
	size_t strides[ndims];

	for (size_t i = 0; i < ndims; i++)
		assert(coords[ndims - i - 1] < d->dims[i]);
	for (size_t i = 0; i < ndims; i++) {
		offsets[i] = coords[ndims - i - 1] * d->tile_dims[i];
		strides[i] = 1;
	}
	for (size_t i = 0; i < ndims; i++)
		dims[i] = (coords[ndims - i - 1] == d->dims[i] - 1 ?
			      d->border_tile_dims[i] :
			      d->tile_dims[i]);

	aml_layout_slice_native(d->layout, &ret, offsets, dims, strides);
	return ret;
}

struct aml_layout*
aml_tiling_resize_row_index_linear(const struct aml_tiling_data *t,
				   const size_t uuid)
{
	(void)t;
	(void)uuid;
	return NULL;
}

int aml_tiling_resize_row_order(const struct aml_tiling_data *t)
{
	(void)t;
	return AML_TILING_ORDER_ROW_MAJOR;
}

int aml_tiling_resize_row_tile_dims(const struct aml_tiling_data *t,
				    size_t *tile_dims)
{
	const struct aml_tiling_resize *d =
	    (const struct aml_tiling_resize *)t;
	assert(d != NULL);
	for (size_t i = 0; i < d->ndims; i++)
		tile_dims[i] = d->tile_dims[d->ndims - i - 1];
	return 0;
}

int aml_tiling_resize_row_dims(const struct aml_tiling_data *t,
			       size_t *dims)
{
	const struct aml_tiling_resize *d =
	    (const struct aml_tiling_resize *)t;
	assert(d != NULL);
	for (size_t i = 0; i < d->ndims; i++)
		dims[i] = d->dims[d->ndims - i - 1];
	return 0;
}

size_t aml_tiling_resize_row_ndims(const struct aml_tiling_data *t)
{
	const struct aml_tiling_resize *d =
	    (const struct aml_tiling_resize *)t;
	assert(d != NULL);
	return d->ndims;
}

size_t aml_tiling_resize_row_ntiles(const struct aml_tiling_data *l)
{
	const struct aml_tiling_resize *d =
	    (const struct aml_tiling_resize *)l;
	assert(d != NULL);
	return 0;
}

struct aml_tiling_ops aml_tiling_resize_row_ops = {
	aml_tiling_resize_row_index,
	aml_tiling_resize_row_index_linear,
	aml_tiling_resize_row_order,
	aml_tiling_resize_row_tile_dims,
	aml_tiling_resize_row_dims,
	aml_tiling_resize_row_ndims,
	aml_tiling_resize_row_ntiles,
};
