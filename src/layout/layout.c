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
aml_check_layout_coords(const struct aml_layout *layout, const ssize_t *coords)
{
	size_t ndims = layout->ops->ndims(layout->data);
	size_t dims[ndims];
	ssize_t bases[ndims];
	int err = AML_SUCCESS;

	err = layout->ops->dims(layout->data, dims);
	if (err != AML_SUCCESS)
		return err;
	err = layout->ops->bases(layout->data, bases);
	if (err != AML_SUCCESS)
		return err;

	for(size_t i = 0; i < ndims; i++) {
		ssize_t max = bases[i] + dims[i];
		if (coords[i] < bases[i] || coords[i] >= max)
			return -AML_EINVAL;
	}

	return AML_SUCCESS;
}

void *aml_layout_deref(const struct aml_layout *layout, const ssize_t *coords)
{
	assert(layout != NULL &&
	       layout->ops != NULL &&
	       layout->ops->deref != NULL);

	return layout->ops->deref(layout->data, coords);
}

void *aml_layout_deref_safe(const struct aml_layout *layout,
			    const ssize_t *coords)
{
	assert(layout != NULL &&
	       layout->ops != NULL &&
	       layout->ops->deref != NULL &&
	       layout->ops->ndims != NULL &&
	       layout->ops->bases != NULL &&
	       layout->ops->dims != NULL);

	assert(aml_check_layout_coords(layout, coords) == AML_SUCCESS);

	return layout->ops->deref(layout->data, coords);
}

void *aml_layout_deref_native(const struct aml_layout *layout,
			      const ssize_t *coords)
{
	assert(layout != NULL &&
	       layout->ops != NULL &&
	       layout->ops->deref_native != NULL &&
	       layout->ops->ndims != NULL &&
	       layout->ops->dims_native != NULL);

	return layout->ops->deref_native(layout->data, coords);
}

void *aml_layout_deref_nobase_native(const struct aml_layout *layout,
			      const ssize_t *coords)
{
	assert(layout != NULL &&
	       layout->ops != NULL &&
	       layout->ops->deref_native != NULL &&
	       layout->ops->ndims != NULL &&
	       layout->ops->dims_native != NULL);

	return layout->ops->deref_nobase_native(layout->data, coords);
}

int aml_layout_order(const struct aml_layout *layout)
{
	assert(layout != NULL &&
	       layout->ops != NULL &&
	       layout->ops->order != NULL);

	return layout->ops->order(layout->data);
}

int aml_layout_bases(const struct aml_layout *layout, ssize_t *bases)
{
	assert(layout != NULL &&
	       layout->ops != NULL &&
	       layout->ops->bases != NULL);

	return layout->ops->bases(layout->data, bases);
}

int aml_layout_bases_native(const struct aml_layout *layout, ssize_t *bases)
{
	assert(layout != NULL &&
	       layout->ops != NULL &&
	       layout->ops->bases_native != NULL);

	return layout->ops->bases_native(layout->data, bases);
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

/* Checks that the offsets don't make the slice overflow out of the layout */
static int
aml_check_layout_slice(const size_t ndims, const size_t *src_dims,
		       const ssize_t *offsets, const size_t *dims,
		       const size_t *strides, const ssize_t *bases)
{
	for (size_t i = 0; i < ndims; i++) {
		if (offsets[i] < bases[i])
			return -AML_EINVAL;

		ssize_t max = bases[i] + src_dims[i];
		ssize_t last_in_slice = offsets[i] + (dims[i]-1) * strides[i];

		if (last_in_slice >= max)
			return -AML_EINVAL;
	}
	return AML_SUCCESS;
}



int aml_layout_slice(const struct aml_layout *layout,
		     struct aml_layout **reshaped_layout,
		     const ssize_t *offsets,
		     const size_t *dims,
		     const size_t *strides)
{
	assert(layout != NULL && layout->ops != NULL);

	if (layout->ops->slice == NULL)
		return -AML_ENOTSUP;

	if (reshaped_layout == NULL)
		return -AML_EINVAL;

	size_t ndims = aml_layout_ndims(layout);
	struct aml_layout *result;
	int err;
	ssize_t _offsets[ndims];
	size_t _dims[ndims];
	size_t _strides[ndims];
	ssize_t bases[ndims];

	assert(layout->ops->bases(layout->data, bases) == AML_SUCCESS);
	assert(layout->ops->dims(layout->data, _dims) == AML_SUCCESS);

	if (offsets)
		memcpy(_offsets, offsets, ndims * sizeof(*offsets));
	else
		for (size_t i = 0; i < ndims; i++)
			_offsets[i] = bases[i];

	if (strides)
		memcpy(_strides, strides, ndims * sizeof(*strides));
	else
		for (size_t i = 0; i < ndims; i++)
			_strides[i] = 1;

	err = aml_check_layout_slice(ndims, _dims, _offsets, dims, _strides,
				      bases);
	if (err != AML_SUCCESS)
		return err;

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
			    const ssize_t *offsets,
			    const size_t *dims,
			    const size_t *strides)
{
	assert(layout != NULL && layout->ops != NULL);

	if (layout->ops->slice_native == NULL)
		return -AML_ENOTSUP;

	assert(layout->ops->ndims != NULL && layout->ops->dims_native != NULL);

	size_t ndims = aml_layout_ndims(layout);
	struct aml_layout *result;
	int err;
	ssize_t _offsets[ndims];
	size_t _dims[ndims];
	size_t _strides[ndims];
	ssize_t bases[ndims];

	assert(layout->ops->bases_native(layout->data, bases) == AML_SUCCESS);
	assert(layout->ops->dims_native(layout->data, _dims) == AML_SUCCESS);

	if (offsets)
		memcpy(_offsets, offsets, ndims * sizeof(*offsets));
	else
		for (size_t i = 0; i < ndims; i++)
			_offsets[i] = bases[i];

	if (strides)
		memcpy(_strides, strides, ndims * sizeof(*strides));
	else
		for (size_t i = 0; i < ndims; i++)
			_strides[i] = 1;


	err = aml_check_layout_slice(ndims, _dims, _offsets, _dims, _strides,
				     bases);
	if (err != AML_SUCCESS)
		return err;

	err = layout->ops->slice_native(layout->data,
					&result, offsets, dims, strides);
	if (err == AML_SUCCESS)
		*reshaped_layout = result;

	return err;
}
