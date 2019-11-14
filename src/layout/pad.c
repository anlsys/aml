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

	layout = AML_INNER_MALLOC_4(struct aml_layout,
				    struct aml_layout_pad,
				    size_t, 2*ndims, element_size);
	if (layout == NULL) {
		*ret = NULL;
		return -AML_ENOMEM;
	}

	data = AML_INNER_MALLOC_NEXTPTR(layout,
					struct aml_layout,
					struct aml_layout_pad);
	layout->data = (struct aml_layout_data *) data;
	data->dims = AML_INNER_MALLOC_EXTRA_NEXTPTR(layout,
						    struct aml_layout,
						    struct aml_layout_pad,
						    size_t, 0);
	data->target_dims = AML_INNER_MALLOC_EXTRA_NEXTPTR(layout,
							struct aml_layout,
							struct aml_layout_pad,
							size_t, ndims);
	data->neutral = AML_INNER_MALLOC_EXTRA_NEXTPTR(layout,
						       struct aml_layout,
						       struct aml_layout_pad,
						       size_t, 2*ndims);
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

void aml_layout_pad_destroy(struct aml_layout **l)
{
	if (l == NULL || *l == NULL)
		return;
	free(*l);
	*l = NULL;
}

/*******************************************************************************
 * COLUMN OPERATORS:
 ******************************************************************************/

void *aml_layout_pad_column_deref(const struct aml_layout_data *data,
				   const ssize_t *coords)
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

struct aml_layout_ops aml_layout_pad_column_ops = {
	aml_layout_pad_column_deref,
	aml_layout_pad_column_deref,
	aml_layout_pad_column_order,
	NULL,
	NULL,
	aml_layout_pad_column_dims,
	aml_layout_pad_column_dims,
	aml_layout_pad_ndims,
	aml_layout_pad_element_size,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
};

/*******************************************************************************
 * ROW OPERATORS:
 ******************************************************************************/

void *aml_layout_pad_row_deref(const struct aml_layout_data *data,
				const ssize_t *coords)
{
	assert(data != NULL);
	const struct aml_layout_pad *d = (const struct aml_layout_pad *)data;
	size_t ndims = d->ndims;
	int type;

	for (size_t i = 0; i < ndims; i++) {
		if (coords[ndims - i - 1]  >= d->target_dims[i])
			return d->neutral;
	}
	type = aml_layout_order(d->target);
	if (AML_LAYOUT_ORDER(type) == AML_LAYOUT_ORDER_ROW_MAJOR) {
		return aml_layout_deref(d->target, coords);
	} else if (AML_LAYOUT_ORDER(type) == AML_LAYOUT_ORDER_COLUMN_MAJOR) {
		ssize_t target_coords[ndims];

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

struct aml_layout_ops aml_layout_pad_row_ops = {
	aml_layout_pad_row_deref,
	aml_layout_pad_column_deref,
	aml_layout_pad_row_order,
	NULL,
	NULL,
	aml_layout_pad_row_dims,
	aml_layout_pad_column_dims,
	aml_layout_pad_ndims,
	aml_layout_pad_element_size,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
};

