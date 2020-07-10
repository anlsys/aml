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

#include "aml/layout/cuda.h"
#include "aml/utils/inner-malloc.h"

int aml_layout_cuda_create(struct aml_layout **out,
                           void *device_ptr,
                           int device,
                           const size_t element_size,
                           const int order,
                           const size_t ndims,
                           const size_t *dims,
                           const size_t *stride,
                           const size_t *pitch)
{
	struct aml_layout *layout;
	struct aml_layout_cuda_data *layout_data;

	layout = AML_INNER_MALLOC_ARRAY(3 * ndims, size_t, struct aml_layout,
	                                struct aml_layout_cuda_data);

	if (layout == NULL)
		return -AML_ENOMEM;

	layout_data = AML_INNER_MALLOC_GET_FIELD(layout, 2, struct aml_layout,
	                                         struct aml_layout_cuda_data);
	layout_data->device_ptr = device_ptr;
	layout_data->device = device;
	layout_data->order = order;
	layout_data->ndims = ndims;
	layout_data->dims = AML_INNER_MALLOC_GET_ARRAY(
	        layout, size_t, struct aml_layout, struct aml_layout_cuda_data);
	layout_data->stride = layout_data->dims + ndims;
	layout_data->cpitch = layout_data->stride + ndims;

	// Store dims, stride and cpitch are internally stored in fortran
	// row major.
	layout_data->cpitch[0] = element_size;
	if (order == AML_LAYOUT_ORDER_COLUMN_MAJOR) {
		layout_data->dims[0] = dims[ndims - 1];
		layout_data->stride[0] = stride[ndims - 1];
		for (size_t i = 1; i < ndims; i++) {
			layout_data->dims[i] = dims[ndims - 1 - i];
			layout_data->stride[i] = stride[ndims - 1 - i];
			layout_data->cpitch[i] = layout_data->cpitch[i - 1] *
			                         pitch[ndims - 1 - i];
		}
	} else {
		memcpy(layout_data->dims, dims, ndims * sizeof(size_t));
		memcpy(layout_data->stride, stride, ndims * sizeof(size_t));
		for (size_t i = 1; i < ndims; i++)
			layout_data->cpitch[i] =
			        layout_data->cpitch[i - 1] * pitch[i];
	}

	layout->data = (struct aml_layout_data *)layout_data;
	layout->ops = &aml_layout_cuda_ops;
	*out = layout;
	return AML_SUCCESS;
}

int aml_layout_cuda_destroy(struct aml_layout **layout)
{
	if (layout == NULL || *layout == NULL)
		return -AML_EINVAL;
	free(*layout);
	*layout = NULL;
	return AML_SUCCESS;
}

void *aml_layout_cuda_deref(const struct aml_layout_data *data,
                            const size_t *coords)
{
	struct aml_layout_cuda_data *cudata;

	(void)coords;
	cudata = (struct aml_layout_cuda_data *)data;
	return cudata->device_ptr;
}

void *aml_layout_cuda_deref_native(const struct aml_layout_data *data,
                                   const size_t *coords)
{
	struct aml_layout_cuda_data *cudata;

	(void)coords;
	cudata = (struct aml_layout_cuda_data *)data;
	return cudata->device_ptr;
}

int aml_layout_cuda_order(const struct aml_layout_data *data)
{
	struct aml_layout_cuda_data *cudata;

	cudata = (struct aml_layout_cuda_data *)data;
	return cudata->order;
}

int aml_layout_cuda_dims(const struct aml_layout_data *data, size_t *dims)
{
	struct aml_layout_cuda_data *cudata;

	cudata = (struct aml_layout_cuda_data *)data;

	if (cudata->order == AML_LAYOUT_ORDER_ROW_MAJOR)
		memcpy(dims, cudata->dims, sizeof(*dims) * cudata->ndims);
	else
		for (size_t i = 0; i < cudata->ndims; i++)
			dims[i] = cudata->dims[cudata->ndims - 1 - i];

	return AML_SUCCESS;
}

int aml_layout_cuda_dims_native(const struct aml_layout_data *data,
                                size_t *dims)
{
	struct aml_layout_cuda_data *cudata;

	cudata = (struct aml_layout_cuda_data *)data;
	memcpy(dims, cudata->dims, sizeof(*dims) * cudata->ndims);
	return AML_SUCCESS;
}

size_t aml_layout_cuda_ndims(const struct aml_layout_data *data)
{
	struct aml_layout_cuda_data *cudata;

	cudata = (struct aml_layout_cuda_data *)data;
	return cudata->ndims;
}

size_t aml_layout_cuda_element_size(const struct aml_layout_data *data)
{
	struct aml_layout_cuda_data *cudata;

	cudata = (struct aml_layout_cuda_data *)data;
	return cudata->cpitch[0];
}

void *aml_layout_cuda_rawptr(const struct aml_layout_data *data)
{
	struct aml_layout_cuda_data *cudata;

	cudata = (struct aml_layout_cuda_data *)data;
	return cudata->device_ptr;
}

struct aml_layout_ops aml_layout_cuda_ops = {
        .deref = aml_layout_cuda_deref,
        .deref_native = aml_layout_cuda_deref_native,
        .rawptr = aml_layout_cuda_rawptr,
        .order = aml_layout_cuda_order,
        .dims = aml_layout_cuda_dims,
        .dims_native = aml_layout_cuda_dims_native,
        .ndims = aml_layout_cuda_ndims,
        .element_size = aml_layout_cuda_element_size,
        .reshape = NULL,
        .slice = NULL,
        .slice_native = NULL,
};
