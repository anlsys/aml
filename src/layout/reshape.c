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

static int aml_layout_reshape_alloc(struct aml_layout **ret,
				    const size_t ndims,
				    const size_t target_ndims)
{
	struct aml_layout *layout;
	struct aml_layout_data_reshape *data;

	layout = AML_INNER_MALLOC_ARRAY(2*ndims + target_ndims, size_t,
					struct aml_layout,
					struct aml_layout_data_reshape);
	if (layout == NULL) {
		*ret = NULL;
		return -AML_ENOMEM;
	}

	data = AML_INNER_MALLOC_GET_FIELD(layout, 2,
					  struct aml_layout,
					  struct aml_layout_data_reshape);

	layout->data = (struct aml_layout_data *)data;

	data->dims = AML_INNER_MALLOC_GET_ARRAY(layout,
						size_t,
						struct aml_layout,
						struct aml_layout_data_reshape);

	data->coffsets = data->dims + ndims;
	data->target_dims = data->coffsets + ndims;

	data->target = NULL;
	data->target_ndims = target_ndims;
	data->ndims = ndims;
	*ret = layout;
	return AML_SUCCESS;
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
	int err;

	if (layout == NULL || target == NULL || ndims == 0)
		return -AML_EINVAL;

	target_ndims = aml_layout_ndims(target);
	err = aml_layout_reshape_alloc(&output, ndims, target_ndims);
	if (err)
		return err;

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

int aml_layout_reshape_duplicate(const struct aml_layout *layout,
                                 struct aml_layout **out)
{
	const struct aml_layout_data_reshape *data;
	struct aml_layout_data_reshape *dret;
	struct aml_layout *ret;
	int err;

	data = (const struct aml_layout_data_reshape *)layout->data;

	if (layout->data == NULL || out == NULL)
		return -AML_EINVAL;

	err = aml_layout_reshape_alloc(&ret, data->ndims, data->target_ndims);
	if (err)
		return err;

	ret->ops = layout->ops;
	dret = (struct aml_layout_data_reshape *)ret->data;
	aml_layout_duplicate(data->target, &dret->target);
	/* small optimization, copying all data at the end of the structure */
	memcpy(dret->dims, data->dims,
	       (2 * data->ndims + data->target_ndims) * sizeof(size_t));
	*out = ret;
	return AML_SUCCESS;
}

void aml_layout_reshape_destroy(struct aml_layout *l)
{
	assert(l != NULL);

	struct aml_layout_data_reshape *data =
	        (struct aml_layout_data_reshape *)l->data;
	aml_layout_destroy(&data->target);
	free(l);
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

void *aml_layout_reshape_rawptr(const struct aml_layout_data *data)
{
	const struct aml_layout_data_reshape *d;

	d = (const struct aml_layout_data_reshape *)data;

	return d->target->ops->rawptr(d->target->data);
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

int aml_layout_reshape_column_fprintf(const struct aml_layout_data *data,
				      FILE *stream, const char *prefix)
{
	const struct aml_layout_data_reshape *d;

	fprintf(stream, "%s: layout-reshape: %p: column-major\n", prefix,
		(void *)data);
	if (data == NULL)
		return AML_SUCCESS;

	d = (const struct aml_layout_data_reshape *)data;

	fprintf(stream, "%s: ndims: %zu\n", prefix, d->ndims);
	fprintf(stream, "%s: target-ndims: %zu\n", prefix, d->target_ndims);
	for (size_t i = 0; i < d->ndims; i++) {
		fprintf(stream, "%s: %16zu: %16zu %16zu\n", prefix, i,
		        d->dims[i], d->coffsets[i]);
	}
	for (size_t i = 0; i < d->target_ndims; i++) {
		fprintf(stream, "%s: %16zu: %16zu\n", prefix, i,
		        d->target_dims[i]);
	}
	fprintf(stream, "%s: target: begin\n", prefix);
	aml_layout_fprintf(stream, prefix, d->target);
	fprintf(stream, "%s: target: end\n", prefix);
	return AML_SUCCESS;
}

struct aml_layout_ops aml_layout_reshape_column_ops = {
        aml_layout_reshape_column_deref,
        aml_layout_reshape_column_deref,
        aml_layout_reshape_rawptr,
        aml_layout_reshape_column_order,
        aml_layout_reshape_column_dims,
        aml_layout_reshape_column_dims,
        aml_layout_reshape_ndims,
        aml_layout_reshape_element_size,
        NULL,
        NULL,
        NULL,
        aml_layout_reshape_column_fprintf,
        aml_layout_reshape_duplicate,
        aml_layout_reshape_destroy,
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

int aml_layout_reshape_row_fprintf(const struct aml_layout_data *data,
				   FILE *stream, const char *prefix)
{
	const struct aml_layout_data_reshape *d;

	fprintf(stream, "%s: layout-reshape: %p: row-major\n", prefix,
		(void *)data);
	if (data == NULL)
		return AML_SUCCESS;

	d = (const struct aml_layout_data_reshape *)data;

	fprintf(stream, "%s: ndims: %zu\n", prefix, d->ndims);
	fprintf(stream, "%s: target-ndims: %zu\n", prefix, d->target_ndims);
	for (size_t i = 0; i < d->ndims; i++) {
		size_t j = d->ndims - i - 1;

		fprintf(stream, "%s: %16zu: %16zu %16zu\n", prefix, i,
		        d->dims[j], d->coffsets[j]);
	}
	for (size_t i = 0; i < d->target_ndims; i++) {
		fprintf(stream, "%s: %16zu: %16zu\n", prefix, i,
		        d->target_dims[i]);
	}

	fprintf(stream, "%s: target: begin\n", prefix);
	aml_layout_fprintf(stream, prefix, d->target);
	fprintf(stream, "%s: target: end\n", prefix);
	return AML_SUCCESS;
}

struct aml_layout_ops aml_layout_reshape_row_ops = {
        aml_layout_reshape_row_deref,
        aml_layout_reshape_column_deref,
        aml_layout_reshape_rawptr,
        aml_layout_reshape_row_order,
        aml_layout_reshape_row_dims,
        aml_layout_reshape_column_dims,
        aml_layout_reshape_ndims,
        aml_layout_reshape_element_size,
        NULL,
        NULL,
        NULL,
        aml_layout_reshape_row_fprintf,
        aml_layout_reshape_duplicate,
        aml_layout_reshape_destroy,
};
