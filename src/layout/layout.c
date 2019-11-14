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

static int
aml_check_layout_coords(const struct aml_layout *layout,
			int (*get_dims)(const struct aml_layout_data *,
					size_t *), const size_t *coords)
{
	size_t ndims = layout->ops->ndims(layout->data);
	size_t dims[ndims];
	int err = AML_SUCCESS;

	err = get_dims(layout->data, dims);
	if (err != AML_SUCCESS)
		return err;

	while (ndims--)
		if (coords[ndims] >= dims[ndims])
			return -AML_EINVAL;

	return AML_SUCCESS;
}

void *aml_layout_deref(const struct aml_layout *layout, const size_t *coords)
{
	assert(layout != NULL &&
	       layout->ops != NULL &&
	       layout->ops->deref != NULL);

	return layout->ops->deref(layout->data, coords);
}

void *aml_layout_deref_safe(const struct aml_layout *layout,
			    const size_t *coords)
{
	assert(layout != NULL &&
	       layout->ops != NULL &&
	       layout->ops->deref != NULL &&
	       layout->ops->ndims != NULL &&
	       layout->ops->dims != NULL);

	assert(aml_check_layout_coords(layout,
				       layout->ops->dims,
				       coords) == AML_SUCCESS);

	return layout->ops->deref(layout->data, coords);
}

void *aml_layout_deref_native(const struct aml_layout *layout,
			      const size_t *coords)
{
	assert(layout != NULL &&
	       layout->ops != NULL &&
	       layout->ops->deref_native != NULL &&
	       layout->ops->ndims != NULL &&
	       layout->ops->dims_native != NULL);

	return layout->ops->deref_native(layout->data, coords);
}

int aml_layout_order(const struct aml_layout *layout)
{
	assert(layout != NULL &&
	       layout->ops != NULL &&
	       layout->ops->order != NULL);

	return layout->ops->order(layout->data);
}

int aml_layout_dims(const struct aml_layout *layout, size_t *dims)
{
	assert(layout != NULL &&
	       layout->ops != NULL &&
	       layout->ops->dims != NULL);

	return layout->ops->dims(layout->data, dims);
}

int aml_layout_dims_native(const struct aml_layout *layout, size_t *dims)
{
	assert(layout != NULL &&
	       layout->ops != NULL &&
	       layout->ops->dims_native != NULL);

	return layout->ops->dims_native(layout->data, dims);
}

size_t aml_layout_ndims(const struct aml_layout *layout)
{
	assert(layout != NULL &&
	       layout->ops != NULL &&
	       layout->ops->ndims != NULL);

	return layout->ops->ndims(layout->data);
}

size_t aml_layout_element_size(const struct aml_layout *layout)
{
	assert(layout != NULL &&
	       layout->ops != NULL &&
	       layout->ops->element_size != NULL);

	return layout->ops->element_size(layout->data);
}

static int aml_layout_check_elements(const struct aml_layout *layout,
				     const size_t ndims,
				     const size_t *dims)
{
	size_t layout_ndims;
	size_t n = 0, m = 0;

	assert(layout->ops->ndims != NULL &&
	       layout->ops->dims != NULL);

	layout_ndims = layout->ops->ndims(layout->data);

	size_t layout_dims[layout_ndims];

	assert(layout->ops->dims(layout->data, layout_dims) == AML_SUCCESS);

	for (size_t i = 0; i < ndims; i++)
		n *= dims[i];

	for (size_t i = 0; i < layout_ndims; i++)
		m *= layout_dims[i];

	if (m != n)
		return -AML_EINVAL;

	return AML_SUCCESS;
}

int aml_layout_reshape(const struct aml_layout *layout,
		       struct aml_layout **output,
		       const size_t ndims,
		       const size_t *dims)
{
	assert(ndims != 0 &&
	       output != NULL &&
	       layout != NULL &&
	       layout->ops != NULL);

	if (layout->ops->reshape == NULL)
		return -AML_ENOTSUP;

	assert(aml_layout_check_elements(layout, ndims, dims) == AML_SUCCESS);

	struct aml_layout *result = NULL;

	int err = layout->ops->reshape(layout->data, &result, ndims, dims);

	if (err == AML_SUCCESS)
		*output = result;

	return err;
}

/**
 * This function will collect the layout dimensions and check that
 * the slice queried will fit into the layout.
 **/
static int
aml_check_layout_slice(const struct aml_layout *layout,
		       int (*get_dims)(const struct aml_layout_data *,
				       size_t *),
		       const size_t *offsets,
		       const size_t *dims,
		       const size_t *strides)
{
	assert(layout->ops->ndims != NULL &&
	       layout->ops->dims != NULL);

	int err = AML_SUCCESS;
	size_t ndims = layout->ops->ndims(layout->data);
	size_t n_elements;
	size_t layout_dims[ndims];

	err = get_dims(layout->data, layout_dims);
	if (err != AML_SUCCESS)
		return err;

	for (size_t i = 0; i < ndims; i++) {
		n_elements = offsets[i] + (dims[i]-1) * strides[i];

		if (n_elements > layout_dims[i])
			return -AML_EINVAL;
	}

	return AML_SUCCESS;
}

int aml_layout_slice(const struct aml_layout *layout,
		     struct aml_layout **reshaped_layout,
		     const size_t *offsets,
		     const size_t *dims,
		     const size_t *strides)
{
	assert(layout != NULL &&
	       layout->ops != NULL);

	if (layout->ops->slice == NULL)
		return -AML_ENOTSUP;

	size_t ndims = aml_layout_ndims(layout);
	struct aml_layout *result;
	int err;
	size_t _offsets[ndims];
	size_t _strides[ndims];

	if (offsets)
		memcpy(_offsets, offsets, ndims * sizeof(*offsets));
	else
		for (size_t i = 0; i < ndims; i++)
			_offsets[i] = 0;

	if (strides)
		memcpy(_strides, strides, ndims * sizeof(*strides));
	else
		for (size_t i = 0; i < ndims; i++)
			_strides[i] = 1;

	assert(aml_check_layout_slice(layout,
				      layout->ops->dims,
				      _offsets,
				      dims,
				      _strides) == AML_SUCCESS);

	err = layout->ops->slice(layout->data,
				 &result,
				 _offsets,
				 dims,
				 _strides);
	if (err == AML_SUCCESS)
		*reshaped_layout = result;

	return err;
}

int aml_layout_slice_native(const struct aml_layout *layout,
			    struct aml_layout **reshaped_layout,
			    const size_t *offsets,
			    const size_t *dims,
			    const size_t *strides)
{
	assert(layout != NULL &&
	       layout->ops != NULL);

	if (layout->ops->slice_native == NULL)
		return -AML_ENOTSUP;

	assert(layout->ops->ndims != NULL &&
	       layout->ops->dims_native != NULL);

	struct aml_layout *result;
	int err;

	assert(aml_check_layout_slice(layout,
				      layout->ops->dims_native,
				      offsets,
				      dims,
				      strides) == AML_SUCCESS);

	err = layout->ops->slice_native(layout->data,
					&result, offsets, dims, strides);
	if (err == AML_SUCCESS)
		*reshaped_layout = result;

	return err;
}
