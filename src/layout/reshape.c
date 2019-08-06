/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
*******************************************************************************/

#include <stdlib.h>

#include "aml.h"
#include "aml/layout/dense.h"
#include "aml/layout/reshape.h"

static struct aml_layout *aml_layout_reshape_alloc(const size_t ndims,
						   const size_t target_ndims)
{
	struct aml_layout *l;
	struct aml_layout_data_reshape *r;
	char *c;

	l = malloc(sizeof(struct aml_layout) +
		   sizeof(struct aml_layout_data_reshape) +
		   2 * ndims * sizeof(size_t) +
		   target_ndims * sizeof(size_t));
	if (l == NULL)
		return NULL;

	c = (char *)l;
	c += sizeof(*l);
	r = (struct aml_layout_data_reshape *)c;
	c += sizeof(*r);
	r->dims = (size_t *)c;
	c += ndims * sizeof(*r->dims);
	r->coffsets = (size_t *)c;
	c += ndims * sizeof(*r->coffsets);
	r->target_dims = (size_t *)c;

	r->target = NULL;
	r->target_ndims = target_ndims;
	r->ndims = ndims;
	l->data = (struct aml_layout_data *) r;
	return l;
}

int aml_layout_reshape_create(struct aml_layout **layout,
			      struct aml_layout *target,
			      const int order,
			      const size_t ndims,
			      const size_t *dims)
{
	struct aml_layout *output;
	struct aml_layout_data_reshape *data;
	size_t target_ndims;
	size_t prod;
	size_t target_prod;

	if (layout == NULL || target == NULL || ndims == 0)
		return -AML_EINVAL;

	target_ndims = aml_layout_ndims(target);
	output = aml_layout_reshape_alloc(ndims, target_ndims);
	if (output == NULL)
		return -AML_ENOMEM;

	data = (struct aml_layout_data_reshape *)output->data;
	data->target = target;

	switch (AML_LAYOUT_ORDER(order)) {
	case AML_LAYOUT_ORDER_ROW_MAJOR:
		output->ops = &aml_layout_reshape_row_ops;
		for (size_t i = 0; i < ndims; i++)
			data->dims[i] = dims[ndims-i-1];
		break;

	case AML_LAYOUT_ORDER_COLUMN_MAJOR:
		output->ops = &aml_layout_reshape_column_ops;
		memcpy(data->dims, dims, ndims * sizeof(size_t));
		break;
	default:
		free(output);
		return -AML_EINVAL;

	}

	size_t target_dims[target_ndims];

	switch (aml_layout_order(target)) {
	case AML_LAYOUT_ORDER_ROW_MAJOR:
		aml_layout_dims(target, target_dims);
		for (size_t i = 0; i < target_ndims; i++)
			data->target_dims[i] = target_dims[target_ndims-i-1];
		break;
	case AML_LAYOUT_ORDER_COLUMN_MAJOR:
		aml_layout_dims(target, data->target_dims);
		break;
	default:
		free(output);
		return -AML_EINVAL;
	}

	prod = 1;
	for (size_t i = 0; i < ndims; i++) {
		data->coffsets[i] = prod;
		prod *= data->dims[i];
	}
	target_prod = 1;
	for (size_t i = 0; i < data->target_ndims; i++)
		target_prod *= data->target_dims[i];

	if (target_prod != prod) {
		free(output);
		return -AML_EINVAL;
	}

	*layout = output;
	return AML_SUCCESS;
}

void aml_layout_reshape_destroy(struct aml_layout **layout)
{
	if (layout == NULL || *layout == NULL)
		return;
	free(*layout);
	*layout = NULL;
}

/*******************************************************************************
 * COLUMN OPERATORS:
 ******************************************************************************/

void *aml_layout_reshape_column_deref(const struct aml_layout_data *data,
				       const size_t *coords)
{
	const struct aml_layout_data_reshape *d;

	d = (const struct aml_layout_data_reshape *)data;

	size_t offset = 0;
	size_t target_coords[d->target_ndims];

	for (size_t i = 0; i < d->ndims; i++)
		offset += coords[i] * d->coffsets[i];

	for (size_t i = 0; i < d->target_ndims; i++) {
		target_coords[i] = offset % d->target_dims[i];
		offset /= d->target_dims[i];
	}

	return d->target->ops->deref_native(d->target->data, target_coords);
}

int aml_layout_reshape_column_order(const struct aml_layout_data *data)
{
	(void) data;
	return AML_LAYOUT_ORDER_COLUMN_MAJOR;
}

int aml_layout_reshape_column_dims(const struct aml_layout_data *data,
				    size_t *dims)
{
	const struct aml_layout_data_reshape *d;

	d = (const struct aml_layout_data_reshape *)data;

	memcpy((void *)dims, (void *)d->dims, sizeof(size_t)*d->ndims);

	return 0;
}

size_t aml_layout_reshape_ndims(const struct aml_layout_data *data)
{
	const struct aml_layout_data_reshape *d;

	d = (const struct aml_layout_data_reshape *)data;

	return d->ndims;
}

size_t aml_layout_reshape_element_size(const struct aml_layout_data *data)
{
	const struct aml_layout_data_reshape *d;

	d = (const struct aml_layout_data_reshape *) data;

	return aml_layout_element_size(d->target);
}

struct aml_layout_ops aml_layout_reshape_column_ops = {
	aml_layout_reshape_column_deref,
	aml_layout_reshape_column_deref,
	aml_layout_reshape_column_order,
	aml_layout_reshape_column_dims,
	aml_layout_reshape_column_dims,
	aml_layout_reshape_ndims,
	aml_layout_reshape_element_size,
	NULL,
	NULL,
	NULL,
};

/*******************************************************************************
 * ROW OPERATORS:
 ******************************************************************************/

void *aml_layout_reshape_row_deref(const struct aml_layout_data *data,
				   const size_t *coords)
{
	const struct aml_layout_data_reshape *d;

	d = (const struct aml_layout_data_reshape *)data;

	size_t offset = 0;
	size_t target_coords[d->target_ndims];

	for (size_t i = 0; i < d->ndims; i++)
		offset += coords[d->ndims - i - 1] * d->coffsets[i];

	for (size_t i = 0; i < d->target_ndims; i++) {
		target_coords[i] = offset % d->target_dims[i];
		offset /= d->target_dims[i];
	}

	return d->target->ops->deref_native(d->target->data, target_coords);
}

int aml_layout_reshape_row_order(const struct aml_layout_data *data)
{
	(void) data;
	return AML_LAYOUT_ORDER_ROW_MAJOR;
}

int aml_layout_reshape_row_dims(const struct aml_layout_data *data,
				size_t *dims)
{
	const struct aml_layout_data_reshape *d;

	d = (const struct aml_layout_data_reshape *)data;

	for (size_t i = 0; i < d->ndims; i++)
		dims[i] = d->dims[d->ndims - i - 1];

	return 0;
}

struct aml_layout_ops aml_layout_reshape_row_ops = {
	aml_layout_reshape_row_deref,
	aml_layout_reshape_column_deref,
	aml_layout_reshape_row_order,
	aml_layout_reshape_row_dims,
	aml_layout_reshape_column_dims,
	aml_layout_reshape_ndims,
	aml_layout_reshape_element_size,
	NULL,
	NULL,
	NULL,
};
