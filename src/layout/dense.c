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
#include "aml/layout/native.h"
#include "aml/layout/dense.h"

static int aml_layout_dense_alloc(struct aml_layout **ret,
				  const size_t ndims)
{
	struct aml_layout *layout;
	struct aml_layout_dense *data;

	layout = AML_INNER_MALLOC_ARRAY(3*ndims, size_t,
					struct aml_layout,
					struct aml_layout_dense);
	if (layout == NULL) {
		*ret = NULL;
		return -AML_ENOMEM;
	}

	data = AML_INNER_MALLOC_GET_FIELD(layout, 2,
					  struct aml_layout,
					  struct aml_layout_dense);
	layout->data = (struct aml_layout_data *) data;

	data->ptr = NULL;
	data->ndims = ndims;

	data->dims = AML_INNER_MALLOC_GET_ARRAY(layout,
						size_t,
						struct aml_layout,
						struct aml_layout_dense);

	data->stride = data->dims + ndims;
	for (size_t i = 0; i < ndims; i++)
		data->stride[i] = 1;

	data->cpitch =  data->stride + ndims;
	*ret = layout;
	return AML_SUCCESS;
}

static
void aml_layout_dense_init_cpitch(struct aml_layout *layout,
				  void *ptr,
				  const size_t ndims,
				  const size_t *dims,
				  const size_t *stride,
				  const size_t *cpitch)
{
	struct aml_layout_dense *data =
		(struct aml_layout_dense *)layout->data;
	data->ptr = ptr;
	memcpy(data->dims, dims, ndims * sizeof(size_t));
	memcpy(data->stride, stride, ndims * sizeof(size_t));
	memcpy(data->cpitch, cpitch, ndims * sizeof(size_t));
}

int aml_layout_dense_create(struct aml_layout **layout,
			    void *ptr,
			    const int order,
			    const size_t element_size,
			    const size_t ndims,
			    const size_t *dims,
			    const size_t *stride,
			    const size_t *pitch)
{

	struct aml_layout *l;
	struct aml_layout_dense *data;
	int err;

	if (layout == NULL || ptr == NULL || !element_size || !ndims ||
	    dims == NULL)
		return -AML_EINVAL;

	err = aml_layout_dense_alloc(&l, ndims);
	if (err)
		return err;

	data = (struct aml_layout_dense *)l->data;
	data->ptr = ptr;
	data->cpitch[0] = element_size;
	size_t _pitch[ndims];

	switch (AML_LAYOUT_ORDER(order)) {

	case AML_LAYOUT_ORDER_ROW_MAJOR:
		l->ops = &aml_layout_row_ops;
		for (size_t i = 0; i < ndims; i++) {
			data->dims[i] = dims[ndims-i-1];
			if (stride)
				data->stride[i] = stride[ndims-i-1];
			if (pitch)
				_pitch[i] = pitch[ndims-i-1];
			else
				_pitch[i] = dims[ndims-i-1];
		}
		break;

	case AML_LAYOUT_ORDER_COLUMN_MAJOR:
		l->ops = &aml_layout_column_ops;
		memcpy(data->dims, dims, ndims * sizeof(size_t));
		if (stride)
			memcpy(data->stride, stride, ndims * sizeof(size_t));
		if (pitch)
			memcpy(_pitch, pitch, ndims * sizeof(size_t));
		else
			memcpy(_pitch, dims, ndims * sizeof(size_t));
		break;
	default:
		free(l);
		return -AML_EINVAL;

	}
	for (size_t i = 1; i < ndims; i++)
		data->cpitch[i] = data->cpitch[i-1]*_pitch[i-1];

	*layout = l;
	return AML_SUCCESS;
}

void aml_layout_dense_destroy(struct aml_layout **l)
{
	if (l == NULL || *l == NULL)
		return;
	free(*l);
	*l = NULL;
}

/*******************************************************************************
 * COLUMN OPERATORS:
 ******************************************************************************/

void *aml_layout_column_deref(const struct aml_layout_data *data,
			      const size_t *coords)
{
	char *ptr;
	const struct aml_layout_dense *d;

	d = (const struct aml_layout_dense *) data;
	ptr = (char *) d->ptr;
	for (size_t i = 0; i < d->ndims; i++)
		ptr += coords[i]*d->cpitch[i]*d->stride[i];
	return (void *)ptr;
}

int aml_layout_column_order(const struct aml_layout_data *data)
{
	(void)data;
	return AML_LAYOUT_ORDER_COLUMN_MAJOR;
}

int aml_layout_column_dims(const struct aml_layout_data *data, size_t *dims)
{
	const struct aml_layout_dense *d;

	d = (const struct aml_layout_dense *) data;
	assert(d != NULL);
	assert(dims != NULL);
	memcpy((void *)dims, (void *)d->dims, sizeof(size_t)*d->ndims);
	return 0;
}

size_t aml_layout_dense_ndims(const struct aml_layout_data *data)
{
	const struct aml_layout_dense *d;

	d = (const struct aml_layout_dense *) data;
	return d->ndims;
}

size_t aml_layout_dense_element_size(const struct aml_layout_data *data)
{
	const struct aml_layout_dense *d;

	d = (const struct aml_layout_dense *)data;
	// element size is the pitch along the 1st dim.
	return d->cpitch[0];
}


/* Given a layout parameters (ndim, dims, stride, and cpitch), returns
 * the representation of this layout that uses the least dimensions.
 * The new parameter are returned in (new_ndims, new_dims, new_stride,
 * and new_cpitch.
 */
static void merge_dims(const size_t ndims,
		       const size_t *dims,
		       const size_t *stride,
		       const size_t *cpitch,
		       size_t *new_ndims,
		       size_t *new_dims,
		       size_t *new_stride,
		       size_t *new_cpitch)
{
	size_t dim_index = 0;
	size_t new_dim_index = 0;

	/* Greedy algorithm that tries to merge dimensions starting with the
	 * first */
	new_dims[new_dim_index] = dims[dim_index];
	new_cpitch[new_dim_index] = cpitch[dim_index];
	new_stride[new_dim_index] = stride[dim_index];
	/* While we haven't consumed all dimensions */
	while (dim_index < ndims - 1) {
		/* Check if current dimension can be merged with the next.
		 * ie: current dimension is not padded && next dimension has
		 * no stride */
		if (dims[dim_index] * stride[dim_index] * cpitch[dim_index] ==
		    cpitch[dim_index + 1] && stride[dim_index + 1] == 1) {
			new_dims[new_dim_index] *= dims[dim_index + 1];
		/* Else add a new dimension with the same characteristic
		 * as the dimensions we were trying to merge */
		} else {
			new_dim_index++;
			new_dims[new_dim_index] = dims[dim_index + 1];
			new_cpitch[new_dim_index] = cpitch[dim_index + 1];
			new_stride[new_dim_index] = stride[dim_index + 1];
		}
		dim_index++;
	}
	new_cpitch[new_dim_index + 1] = 0;
	*new_ndims = new_dim_index + 1;
}


/* Try to change the indexing dimensions of a layout to the
 * given number of dimensions and dimensions. If the description
 * is incompatible with the original layout returns -AML_EINVAL.
 * Else returns AML_SUCCESS as well as new stride and cumulative pitch
 * in n_stride and n_cpitch respectively.
 */
static int reshape_dims(const struct aml_layout_dense *d,
			 const size_t ndims,
			 const size_t *dims,
			 size_t *n_stride,
			 size_t *n_cpitch)
{
	size_t m_ndims;
	size_t m_dims[d->ndims];
	size_t m_stride[d->ndims];
	/* for simplicity, the underlying algorithm needs one more slot */
	size_t m_cpitch[d->ndims + 1];

	/* First obtain a canonical representation of the layout
	 * that uses the least amount of dimensions. */
	merge_dims(d->ndims, d->dims, d->stride, d->cpitch,
		   &m_ndims, m_dims, m_stride, m_cpitch);

	size_t m_dim_index = 0;

	/* Greedy algorithm that tries to split the canonical
	 * representation into the given new dimensions starting from the
	 * first. The canonical representation is destroyed in the process. */
	n_cpitch[0] = m_cpitch[m_dim_index];
	for (size_t i = 0; i < ndims; i++) {
		/* If the new dimension perfectly fits in the current merged
		 * dimensions, then the new stride and cumulative pitch are
		 * copied from the current merged dimension. The next merged
		 * dimension becomes current. */
		if (m_dims[m_dim_index] == dims[i]) {
			n_stride[i] = m_stride[m_dim_index];
			n_cpitch[i + 1] = m_cpitch[m_dim_index + 1];
			m_dim_index++;
		/* Else if the current merged dimension can be evenly split by
		 * the new dimension, we divide the current merged dimension by
		 * the new dimension, merged stride is consumed and becomes 1
		 * and cumulative pitch is computed from the new stride and
		 * dimensions. */
		} else if (m_dims[m_dim_index] % dims[i] == 0) {
			m_dims[m_dim_index] /= dims[i];
			n_stride[i] = m_stride[m_dim_index];
			n_cpitch[i + 1] =
				n_cpitch[i] * dims[i] * m_stride[m_dim_index];
			m_stride[m_dim_index] = 1;
		/* Else the new description is incompatible. */
		} else {
			return -AML_EINVAL;
		}
	}
	return AML_SUCCESS;
}

int aml_layout_column_reshape(const struct aml_layout_data *data,
			      struct aml_layout **output,
			      size_t ndims,
			      const size_t *dims)
{
	int err;
	struct aml_layout *layout;
	const struct aml_layout_dense *d;
	size_t stride[ndims];
	/* for simplicity, the underlying algorithm needs one more slot */
	size_t cpitch[ndims + 1];

	d = (const struct aml_layout_dense *)data;

	err = aml_layout_dense_alloc(&layout, ndims);
	if (err)
		return err;

	err = reshape_dims(d, ndims, dims, stride, cpitch);
	if (err != AML_SUCCESS) {
		free(layout);
		return err;
	}

	aml_layout_dense_init_cpitch(layout,
				     d->ptr,
				     ndims,
				     dims,
				     stride,
				     cpitch);
	layout->ops = &aml_layout_column_ops;

	*output = layout;
	return AML_SUCCESS;
}

int aml_layout_column_slice(const struct aml_layout_data *data,
			    struct aml_layout **output,
			    const size_t *offsets,
			    const size_t *dims,
			    const size_t *strides)
{
	struct aml_layout *layout;
	const struct aml_layout_dense *d;
	void *ptr;
	int err;

	d = (const struct aml_layout_dense *)data;
	ptr = aml_layout_column_deref(data, offsets);

	err = aml_layout_dense_alloc(&layout, d->ndims);
	if (err)
		return err;

	size_t cpitch[d->ndims];
	size_t new_strides[d->ndims];

	for (size_t i = 0; i < d->ndims; i++) {
		cpitch[i] = d->cpitch[i];
		new_strides[i] = strides[i] * d->stride[i];
	}

	aml_layout_dense_init_cpitch(layout,
				     ptr,
				     d->ndims,
				     dims,
				     new_strides,
				     cpitch);
	layout->ops = &aml_layout_column_ops;

	*output = layout;
	return AML_SUCCESS;
}

struct aml_layout_ops aml_layout_column_ops = {
	aml_layout_column_deref,
	aml_layout_column_deref,
	aml_layout_column_order,
	aml_layout_column_dims,
	aml_layout_column_dims,
	aml_layout_dense_ndims,
	aml_layout_dense_element_size,
	aml_layout_column_reshape,
	aml_layout_column_slice,
	aml_layout_column_slice,
};

/*******************************************************************************
 * ROW OPERATORS:
 ******************************************************************************/

void *aml_layout_row_deref(const struct aml_layout_data *data,
			   const size_t *coords)
{
	const struct aml_layout_dense *d;
	char *ptr;

	d = (const struct aml_layout_dense *)data;
	ptr = (char *) d->ptr;

	for (size_t i = 0; i < d->ndims; i++) {
		ptr +=
			coords[i] *
			d->cpitch[d->ndims - i - 1] *
			d->stride[d->ndims - i - 1];
	}
	return (void *) ptr;
}

int aml_layout_row_order(const struct aml_layout_data *data)
{
	(void) data;
	return AML_LAYOUT_ORDER_ROW_MAJOR;
}

int aml_layout_row_dims(const struct aml_layout_data *data, size_t *dims)
{
	const struct aml_layout_dense *d;

	d = (const struct aml_layout_dense *)data;
	for (size_t i = 0; i < d->ndims; i++)
		dims[i] = d->dims[d->ndims - i - 1];
	return 0;
}

int aml_layout_row_reshape(const struct aml_layout_data *data,
			   struct aml_layout **output,
			   const size_t ndims,
			   const size_t *dims)
{
	struct aml_layout *layout;
	const struct aml_layout_dense *d;
	size_t stride[ndims];
	/* for simplicity, the underlying algorithm needs one more slot */
	size_t cpitch[ndims + 1];
	size_t n_dims[ndims];
	int err;

	d = (const struct aml_layout_dense *)data;
	err = aml_layout_dense_alloc(&layout, ndims);
	if (err)
		return err;

	for (size_t i = 0; i < ndims; i++)
		n_dims[ndims - i - 1] = dims[i];

	err = reshape_dims(d, ndims, n_dims, stride, cpitch);
	if (err != AML_SUCCESS) {
		free(layout);
		return err;
	}

	aml_layout_dense_init_cpitch(layout,
				     d->ptr,
				     ndims,
				     n_dims,
				     stride,
				     cpitch);

	layout->ops = &aml_layout_row_ops;
	*output = layout;
	return AML_SUCCESS;
}

int aml_layout_row_slice(const struct aml_layout_data *data,
			 struct aml_layout **output,
			 const size_t *offsets,
			 const size_t *dims,
			 const size_t *strides)
{
	struct aml_layout *layout;
	const struct aml_layout_dense *d;
	void *ptr;
	int err;

	d = (const struct aml_layout_dense *)data;

	size_t cpitch[d->ndims];
	size_t n_offsets[d->ndims];
	size_t n_dims[d->ndims];
	size_t n_strides[d->ndims];

	err = aml_layout_dense_alloc(&layout, d->ndims);
	if (err)
		return err;

	for (size_t i = 0; i < d->ndims; i++) {
		n_offsets[i] = offsets[d->ndims - i - 1];
		n_dims[i] = dims[d->ndims - i - 1];
		n_strides[i] = strides[d->ndims - i - 1];
	}

	for (size_t i = 0; i < d->ndims; i++) {
		cpitch[i] = d->cpitch[i];
		n_strides[i] *= d->stride[i];
	}

	ptr = aml_layout_column_deref(data, n_offsets);
	aml_layout_dense_init_cpitch(layout,
				     ptr,
				     d->ndims,
				     n_dims,
				     n_strides,
				     cpitch);
	layout->ops = &aml_layout_row_ops;

	*output = layout;
	return AML_SUCCESS;
}

int aml_layout_row_slice_native(const struct aml_layout_data *data,
				struct aml_layout **output,
				const size_t *offsets,
				const size_t *dims,
				const size_t *strides)
{
	struct aml_layout *layout;
	const struct aml_layout_dense *d;
	void *ptr;
	int err;

	d = (const struct aml_layout_dense *)data;

	size_t cpitch[d->ndims];
	size_t new_strides[d->ndims];

	err = aml_layout_dense_alloc(&layout, d->ndims);
	if (err)
		return err;

	for (size_t i = 0; i < d->ndims; i++) {
		cpitch[i] = d->cpitch[i];
		new_strides[i] = strides[i] * d->stride[i];
	}

	ptr = aml_layout_column_deref(data, offsets);
	aml_layout_dense_init_cpitch(layout,
				     ptr,
				     d->ndims,
				     dims,
				     new_strides,
				     cpitch);
	layout->ops = &aml_layout_row_ops;
	*output = layout;

	return AML_SUCCESS;
}

struct aml_layout_ops aml_layout_row_ops = {
	aml_layout_row_deref,
	aml_layout_column_deref,
	aml_layout_row_order,
	aml_layout_row_dims,
	aml_layout_column_dims,
	aml_layout_dense_ndims,
	aml_layout_dense_element_size,
	aml_layout_row_reshape,
	aml_layout_row_slice,
	aml_layout_row_slice_native
};

