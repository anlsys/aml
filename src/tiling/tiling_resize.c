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

	tiling = AML_INNER_MALLOC_ARRAY(3*ndims, size_t,
					struct aml_tiling,
					struct aml_tiling_resize);

	if (tiling == NULL) {
		*ret = NULL;
		return -AML_ENOMEM;
	}

	data = AML_INNER_MALLOC_GET_FIELD(tiling,
					  2,
					  struct aml_tiling,
					  struct aml_tiling_resize);
	tiling->data = (struct aml_tiling_data *)data;

	data->tile_dims = AML_INNER_MALLOC_GET_ARRAY(tiling,
						     size_t,
						     struct aml_tiling,
						     struct aml_tiling_resize);

	data->dims = data->tile_dims + ndims;
	data->border_tile_dims = data->dims + ndims;

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

void aml_tiling_resize_destroy(struct aml_tiling **tiling)
{
	if (tiling == NULL)
		return;
	free(*tiling);
	*tiling = NULL;
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

void *aml_tiling_resize_column_rawptr(const struct aml_tiling_data *t,
				      const size_t *coords)
{
	const struct aml_tiling_resize *d =
	    (const struct aml_tiling_resize *)t;

	assert(d != NULL);
	size_t ndims = d->ndims;
	size_t offsets[ndims];

	for (size_t i = 0; i < ndims; i++) {
		assert(coords[i] < d->dims[i]);
		offsets[i] = coords[i] * d->tile_dims[i];
	}

	return aml_layout_deref_native(d->layout, offsets);
}

int aml_tiling_resize_column_tileid(const struct aml_tiling_data *t,
				    const size_t *coords)
{
	const struct aml_tiling_resize *d =
		(const struct aml_tiling_resize *)t;

	assert(d != NULL);
	int ret = 0;
	size_t ndims = d->ndims;

	for (size_t i = 0; i < ndims; i++)
		ret = (ret * d->dims[i]) + coords[i];
	return ret;
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

	size_t ret = 1;

	for (size_t i = 0; i < d->ndims; i++)
		ret = ret * d->dims[i];
	return ret;
}

int aml_tiling_resize_column_fprintf(const struct aml_tiling_data *data,
				     FILE *stream, const char *prefix)
{
	const struct aml_tiling_resize *d;

	fprintf(stream, "%s: tiling-resize: %p: column-major\n", prefix,
		(void *)data);
	if (data == NULL)
		return AML_SUCCESS;

	d = (const struct aml_tiling_resize *)data;

	fprintf(stream, "%s: tags: %d\n", prefix, d->tags);
	fprintf(stream, "%s: ndims: %zu\n", prefix, d->ndims);
	for (size_t i = 0; i < d->ndims; i++) {
		fprintf(stream, "%s: %16zu: %16zu %16zu %16zu\n", prefix,
			i, d->dims[i], d->tile_dims[i], d->border_tile_dims[i]);
	}
	fprintf(stream, "%s: layout: begin\n", prefix);
	aml_layout_fprintf(stream, prefix, d->layout);
	fprintf(stream, "%s: layout: end\n", prefix);
	return AML_SUCCESS;
}

struct aml_tiling_ops aml_tiling_resize_column_ops = {
	aml_tiling_resize_column_index,
	aml_tiling_resize_column_index,
	aml_tiling_resize_column_rawptr,
	aml_tiling_resize_column_tileid,
	aml_tiling_resize_column_order,
	aml_tiling_resize_column_tile_dims,
	aml_tiling_resize_column_dims,
	aml_tiling_resize_column_dims,
	aml_tiling_resize_column_ndims,
	aml_tiling_resize_column_ntiles,
	aml_tiling_resize_column_fprintf,
};

/*******************************************************************************
 * Row Implementation
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

void *aml_tiling_resize_row_rawptr(const struct aml_tiling_data *t,
				      const size_t *coords)
{
	const struct aml_tiling_resize *d =
	    (const struct aml_tiling_resize *)t;

	assert(d != NULL);
	size_t ndims = d->ndims;
	size_t offsets[ndims];

	for (size_t i = 0; i < ndims; i++) {
		assert(coords[ndims - i - 1] < d->dims[i]);
		offsets[i] = coords[ndims - i - 1] * d->tile_dims[i];
	}

	return aml_layout_deref_native(d->layout, offsets);
}

int aml_tiling_resize_row_tileid(const struct aml_tiling_data *t,
				 const size_t *coords)
{
	const struct aml_tiling_resize *d =
		(const struct aml_tiling_resize *)t;

	assert(d != NULL);
	int ret = 0;
	size_t ndims = d->ndims;

	for (size_t i = 0; i < ndims; i++)
		ret = (ret * d->dims[i]) + coords[ndims - i - 1];
	return ret;
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

int aml_tiling_resize_row_fprintf(const struct aml_tiling_data *data,
				  FILE *stream, const char *prefix)
{
	const struct aml_tiling_resize *d;

	fprintf(stream, "%s: tiling-resize: %p: row-major\n", prefix,
		(void *)data);
	if (data == NULL)
		return AML_SUCCESS;

	d = (const struct aml_tiling_resize *)data;

	fprintf(stream, "%s: tags: %d\n", prefix, d->tags);
	fprintf(stream, "%s: ndims: %zu\n", prefix, d->ndims);
	for (size_t i = 0; i < d->ndims; i++) {
		size_t j = d->ndims - i - 1;

		fprintf(stream, "%s: %16zu: %16zu %16zu %16zu\n", prefix,
			i, d->dims[j], d->tile_dims[j], d->border_tile_dims[j]);
	}
	fprintf(stream, "%s: layout: begin\n", prefix);
	aml_layout_fprintf(stream, prefix, d->layout);
	fprintf(stream, "%s: layout: end\n", prefix);
	return AML_SUCCESS;
}

struct aml_tiling_ops aml_tiling_resize_row_ops = {
	aml_tiling_resize_row_index,
	aml_tiling_resize_column_index,
	aml_tiling_resize_row_rawptr,
	aml_tiling_resize_row_tileid,
	aml_tiling_resize_row_order,
	aml_tiling_resize_row_tile_dims,
	aml_tiling_resize_row_dims,
	aml_tiling_resize_column_dims,
	aml_tiling_resize_row_ndims,
	aml_tiling_resize_column_ntiles,
	aml_tiling_resize_row_fprintf,
};
