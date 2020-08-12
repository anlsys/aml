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
#include "aml/layout/pad.h"

static int aml_layout_pad_alloc(struct aml_layout **ret,
				const size_t ndims, size_t element_size)
{
	struct aml_layout *layout;
	struct aml_layout_pad *data;

	layout = AML_INNER_MALLOC_EXTRA(2*ndims, size_t,
					element_size,
					struct aml_layout,
					struct aml_layout_pad);
	if (layout == NULL) {
		*ret = NULL;
		return -AML_ENOMEM;
	}

	data = AML_INNER_MALLOC_GET_FIELD(layout, 2,
					  struct aml_layout,
					  struct aml_layout_pad);
	layout->data = (struct aml_layout_data *) data;

	data->dims = AML_INNER_MALLOC_GET_ARRAY(layout,
						size_t,
						struct aml_layout,
						struct aml_layout_pad);
	data->target_dims = data->dims + ndims;

	data->neutral = AML_INNER_MALLOC_GET_EXTRA(layout,
						   2*ndims, size_t,
						   struct aml_layout,
						   struct aml_layout_pad);
	data->target = NULL;
	data->ndims = ndims;
	data->element_size = element_size;
	*ret = layout;
	return AML_SUCCESS;
}

int aml_layout_pad_create(struct aml_layout **layout, const int order,
			  struct aml_layout *target, const size_t *dims,
			  void *neutral)
{
	struct aml_layout *l;
	struct aml_layout_pad *data;
	int err, type;
	size_t ndims, element_size;

	if (layout == NULL || target == NULL || dims == NULL || neutral == NULL)
		return -AML_EINVAL;

	ndims = aml_layout_ndims(target);
	element_size = aml_layout_element_size(target);

	if (!ndims || !element_size)
		return -AML_EINVAL;

	err = aml_layout_pad_alloc(&l, ndims, element_size);
	if (err)
		return err;

	data = (struct aml_layout_pad *)l->data;
	data->target = target;
	memcpy(data->neutral, neutral, element_size);
	data->tags = order;
	switch (AML_LAYOUT_ORDER(order)) {
	case AML_LAYOUT_ORDER_ROW_MAJOR:
		l->ops = &aml_layout_pad_row_ops;
		for (size_t i = 0; i < ndims; i++)
			data->dims[i] = dims[ndims-i-1];
		break;
	case AML_LAYOUT_ORDER_COLUMN_MAJOR:
		l->ops = &aml_layout_pad_column_ops;
		memcpy(data->dims, dims, ndims * sizeof(size_t));
		break;
	default:
		free(l);
		return -AML_EINVAL;

	}
	type = aml_layout_order(target);
	if (AML_LAYOUT_ORDER(type) == AML_LAYOUT_ORDER_ROW_MAJOR) {
		size_t target_dims[ndims];

		aml_layout_dims(target, target_dims);
		for (size_t i = 0; i < ndims; i++)
			data->target_dims[i] = target_dims[ndims-i-1];
	} else if (AML_LAYOUT_ORDER(type) == AML_LAYOUT_ORDER_COLUMN_MAJOR) {
		aml_layout_dims(target, data->target_dims);
	} else {
		free(l);
		return -AML_EINVAL;
	}
	*layout = l;
	return AML_SUCCESS;
}

int aml_layout_pad_duplicate(const struct aml_layout *layout,
                             struct aml_layout **out)
{
	const struct aml_layout_pad *data;
	struct aml_layout_pad *dret;
	struct aml_layout *ret;
	size_t sz;
	int err;

	data = (const struct aml_layout_pad *)layout->data;

	if (layout->data == NULL || out == NULL)
		return -AML_EINVAL;

	err = aml_layout_pad_alloc(&ret, data->ndims, data->element_size);
	if (err)
		return err;

	ret->ops = layout->ops;
	dret = (struct aml_layout_pad *)ret->data;
	aml_layout_duplicate(data->target, &dret->target);
	dret->tags = data->tags;
	/* small optimization to copy everything at the end of our single
	 * allocation, but careful about neutral and the arrays having a gap
	 **/
	sz = ((char *)dret->neutral - (char *)dret->dims) + data->element_size;
	memcpy(dret->dims, data->dims, sz);
	*out = ret;
	return AML_SUCCESS;
}

void aml_layout_pad_destroy(struct aml_layout *l)
{
	assert(l != NULL);

	struct aml_layout_pad *data = (struct aml_layout_pad *)l->data;
	aml_layout_destroy(&data->target);
	free(l);
}

/*******************************************************************************
 * COLUMN OPERATORS:
 ******************************************************************************/

void *aml_layout_pad_column_deref(const struct aml_layout_data *data,
				   const size_t *coords)
{
	assert(data != NULL);
	const struct aml_layout_pad *d = (const struct aml_layout_pad *)data;
	size_t ndims = d->ndims;

	for (size_t i = 0; i < ndims; i++) {
		if (coords[i] >= d->target_dims[i])
			return d->neutral;
	}
	return d->target->ops->deref_native(d->target->data, coords);
}

void *aml_layout_pad_rawptr(const struct aml_layout_data *data)
{
	const struct aml_layout_pad *d;

	d = (const struct aml_layout_pad *)data;

	return d->target->ops->rawptr(d->target->data);
}

int aml_layout_pad_column_order(const struct aml_layout_data *data)
{
	(void)data;
	return AML_LAYOUT_ORDER_COLUMN_MAJOR;
}

int aml_layout_pad_column_dims(const struct aml_layout_data *data,
			       size_t *dims)
{
	assert(data != NULL);
	assert(dims != NULL);
	const struct aml_layout_pad *d = (const struct aml_layout_pad *)data;

	memcpy((void *)dims, (void *)d->dims, sizeof(size_t)*d->ndims);
	return 0;
}

size_t aml_layout_pad_ndims(const struct aml_layout_data *data)
{
	const struct aml_layout_pad *d = (const struct aml_layout_pad *)data;

	return d->ndims;
}

size_t aml_layout_pad_element_size(const struct aml_layout_data *data)
{
	const struct aml_layout_pad *d = (const struct aml_layout_pad *)data;

	return d->element_size;
}

int aml_layout_pad_column_fprintf(const struct aml_layout_data *data,
				  FILE *stream, const char *prefix)
{
	const struct aml_layout_pad *d;

	fprintf(stream, "%s: layout-pad: %p: column-major\n", prefix,
		(void *)data);
	if (data == NULL)
		return AML_SUCCESS;

	d = (const struct aml_layout_pad *)data;

	fprintf(stream, "%s: element size: %zu\n", prefix, d->element_size);
	fprintf(stream, "%s: neutral: %p\n", prefix, d->neutral);
	fprintf(stream, "%s: ndims: %zu\n", prefix, d->ndims);
	for (size_t i = 0; i < d->ndims; i++) {
		fprintf(stream, "%s: %16zu: %16zu %16zu\n", prefix,
			i, d->dims[i], d->target_dims[i]);
	}
	fprintf(stream, "%s: target: begin\n", prefix);
	aml_layout_fprintf(stream, prefix, d->target);
	fprintf(stream, "%s: target: end\n", prefix);
	return AML_SUCCESS;
}

struct aml_layout_ops aml_layout_pad_column_ops = {
        aml_layout_pad_column_deref,
        aml_layout_pad_column_deref,
        aml_layout_pad_rawptr,
        aml_layout_pad_column_order,
        aml_layout_pad_column_dims,
        aml_layout_pad_column_dims,
        aml_layout_pad_ndims,
        aml_layout_pad_element_size,
        NULL,
        NULL,
        NULL,
        aml_layout_pad_column_fprintf,
        aml_layout_pad_duplicate,
        aml_layout_pad_destroy,
};

/*******************************************************************************
 * ROW OPERATORS:
 ******************************************************************************/

void *aml_layout_pad_row_deref(const struct aml_layout_data *data,
				  const size_t *coords)
{
	assert(data != NULL);
	const struct aml_layout_pad *d = (const struct aml_layout_pad *)data;
	size_t ndims = d->ndims;
	int type;

	for (size_t i = 0; i < ndims; i++) {
		if (coords[ndims - i - 1] >= d->target_dims[i])
			return d->neutral;
	}
	type = aml_layout_order(d->target);
	if (AML_LAYOUT_ORDER(type) == AML_LAYOUT_ORDER_ROW_MAJOR) {
		return aml_layout_deref(d->target, coords);
	} else if (AML_LAYOUT_ORDER(type) == AML_LAYOUT_ORDER_COLUMN_MAJOR) {
		size_t target_coords[ndims];

		for (size_t i = 0; i < ndims; i++)
			target_coords[i] = coords[ndims - i - 1];
		return aml_layout_deref(d->target, target_coords);
	} else
		return NULL;
}

int aml_layout_pad_row_order(const struct aml_layout_data *data)
{
	(void)data;
	return AML_LAYOUT_ORDER_ROW_MAJOR;
}

int aml_layout_pad_row_dims(const struct aml_layout_data *data, size_t *dims)
{
	assert(data != NULL);
	const struct aml_layout_pad *d = (const struct aml_layout_pad *)data;

	for (size_t i = 0; i < d->ndims; i++)
		dims[i] = d->dims[d->ndims - i - 1];
	return 0;
}

int aml_layout_pad_row_fprintf(const struct aml_layout_data *data,
			       FILE *stream, const char *prefix)
{
	const struct aml_layout_pad *d;

	fprintf(stream, "%s: layout-pad: %p: row-major\n", prefix,
		(void *)data);
	if (data == NULL)
		return AML_SUCCESS;

	d = (const struct aml_layout_pad *)data;

	fprintf(stream, "%s: element size: %zu\n", prefix, d->element_size);
	fprintf(stream, "%s: neutral: %p\n", prefix, d->neutral);
	fprintf(stream, "%s: ndims: %zu\n", prefix, d->ndims);
	for (size_t i = 0; i < d->ndims; i++) {
		size_t j = d->ndims - i - 1;

		fprintf(stream, "%s: %16zu: %16zu %16zu\n", prefix,
			i, d->dims[j], d->target_dims[j]);
	}
	fprintf(stream, "%s: target: begin\n", prefix);
	aml_layout_fprintf(stream, prefix, d->target);
	fprintf(stream, "%s: target: end\n", prefix);
	return AML_SUCCESS;
}

struct aml_layout_ops aml_layout_pad_row_ops = {
        aml_layout_pad_row_deref,
        aml_layout_pad_column_deref,
        aml_layout_pad_rawptr,
        aml_layout_pad_row_order,
        aml_layout_pad_row_dims,
        aml_layout_pad_column_dims,
        aml_layout_pad_ndims,
        aml_layout_pad_element_size,
        NULL,
        NULL,
        NULL,
        aml_layout_pad_row_fprintf,
        aml_layout_pad_duplicate,
        aml_layout_pad_destroy,
};
