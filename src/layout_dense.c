#include <aml.h>

/*******************************************************************************
 * Native layout initialization:
 ******************************************************************************/

int aml_layout_native_struct_init(struct aml_layout *layout, size_t ndims,
				  void *memory)
{
	struct aml_layout_data_native *dataptr;

	assert(layout == (struct aml_layout *)memory);
	memory = (void *)((uintptr_t)memory +
		      sizeof(struct aml_layout));
	dataptr = memory;
	layout->data = memory;
	memory = (void *)((uintptr_t)memory +
		      sizeof(struct aml_layout_data_native));
	dataptr->ndims = ndims;
	dataptr->dims = (size_t *)memory;
	dataptr->stride = dataptr->dims + ndims;
	dataptr->pitch = dataptr->stride + ndims;
	dataptr->cpitch = dataptr->pitch + ndims;
	return 0;
}

int aml_layout_native_ainit(struct aml_layout *layout, uint64_t tags, void *ptr,
			    const size_t element_size, size_t ndims,
			    const size_t *dims, const size_t *stride,
			    const size_t *pitch)
{
	assert(layout != NULL);
	assert(layout->data != NULL);
	struct aml_layout_data_native *data =
	    (struct aml_layout_data_native *)layout->data;
	assert(data->ndims == ndims);
	assert(data->dims);
	assert(data->stride);
	assert(data->pitch);
	assert(data->cpitch);
	data->ptr = ptr;
	int type = AML_TYPE_GET(tags, AML_TYPE_LAYOUT_ORDER);
	if(type == AML_TYPE_LAYOUT_ROW_ORDER)
	{
		AML_TYPE_SET(layout->tags, AML_TYPE_LAYOUT_ROW_ORDER);
		layout->ops = &aml_layout_row_ops;
		for(size_t i = 0; i < ndims; i++)
		{
			data->dims[i] = dims[ndims-i-1];
			data->stride[i] = stride[ndims-i-1];
			data->pitch[i] = pitch[ndims-i-1];
		}
		data->cpitch[0] = element_size;
		for(size_t i = 1; i < ndims; i++)
			data->cpitch[i] = data->cpitch[i-1]*pitch[ndims-i];
	}
	else if(type == AML_TYPE_LAYOUT_COLUMN_ORDER)
	{
		AML_TYPE_SET(layout->tags, AML_TYPE_LAYOUT_COLUMN_ORDER);
		layout->ops = &aml_layout_column_ops;
		memcpy(data->dims, dims, ndims * sizeof(size_t));
		memcpy(data->stride, stride, ndims * sizeof(size_t));
		memcpy(data->pitch, pitch, ndims * sizeof(size_t));
		/* pitches are only necessary for ndims-1 dimensions. Since we
		 * store element size as p->cpitch[0], there's still ndims
		 * elements in the array.
		 */
		data->cpitch[0] = element_size;
		for(size_t i = 1; i < ndims; i++)
			data->cpitch[i] = data->cpitch[i-1]*pitch[i-1];
	}
	return 0;
}

int aml_layout_native_vinit(struct aml_layout *p, uint64_t tags, void *ptr,
			    const size_t element_size, size_t ndims, va_list ap)
{
	size_t dims[ndims];
	size_t stride[ndims];
	size_t pitch[ndims-1];
	for(size_t i = 0; i < ndims; i++)
		dims[i] = va_arg(ap, size_t);
	for(size_t i = 0; i < ndims; i++)
		stride[i] = va_arg(ap, size_t);
	for(size_t i = 0; i < ndims; i++)
		pitch[i] = va_arg(ap, size_t);
	return aml_layout_native_ainit(p, tags, ptr, element_size, ndims, dims,
				       stride, pitch);
}

int aml_layout_native_init(struct aml_layout *p, uint64_t tags, void *ptr,
			   const size_t element_size, size_t ndims, ...)
{
	int err;
	va_list ap;
	va_start(ap, ndims);
	err = aml_layout_native_vinit(p, tags, ptr, element_size, ndims, ap);
	va_end(ap);
	return err;
}

int aml_layout_native_acreate(struct aml_layout **layout, uint64_t tags,
			      void *ptr, const size_t element_size,
			      size_t ndims, const size_t *dims,
			      const size_t *stride, const size_t *pitch)
{
	assert(ndims > 0);
	void *baseptr = calloc(1, AML_LAYOUT_NATIVE_ALLOCSIZE(ndims));
	*layout = (struct aml_layout *)baseptr;
	aml_layout_native_struct_init(*layout, ndims, baseptr);
	return aml_layout_native_ainit(*layout, tags, ptr, element_size, ndims,
				       dims, stride, pitch);
}

int aml_layout_native_vcreate(struct aml_layout **layout, uint64_t tags,
			      void *ptr, const size_t element_size,
			      size_t ndims, va_list ap)
{
	assert(ndims > 0);
	void *baseptr = calloc(1, AML_LAYOUT_NATIVE_ALLOCSIZE(ndims));
	*layout = (struct aml_layout *)baseptr;
	aml_layout_native_struct_init(*layout, ndims, baseptr);
	return aml_layout_native_vinit(*layout, tags, ptr, element_size, ndims,
				       ap);
}

int aml_layout_native_create(struct aml_layout **layout, uint64_t tags,
			     void *ptr, const size_t element_size, size_t ndims,
			     ...)
{
	int err;
	va_list ap;
	assert(ndims > 0);
	void *baseptr = calloc(1, AML_LAYOUT_NATIVE_ALLOCSIZE(ndims));
	*layout = (struct aml_layout *)baseptr;
	aml_layout_native_struct_init(*layout, ndims, baseptr);
	va_start(ap, ndims);
	err = aml_layout_native_vinit(*layout, tags, ptr, element_size, ndims,
				      ap);
	va_end(ap);
	return err;
}

/*******************************************************************************
 * COLUMN OPERATORS:
 ******************************************************************************/

void *aml_layout_column_deref(const struct aml_layout_data *data,
			      va_list coords)
{
	const struct aml_layout_data_native *d =
	    (const struct aml_layout_data_native *)data;
	void *ptr;
	assert(d != NULL);
	assert(d->ptr != NULL);
	ptr = d->ptr;
	for(size_t i = 0; i < d->ndims; i++)
	{
		size_t c = va_arg(coords, size_t);
		assert(c < d->dims[i]);
		ptr += c*d->cpitch[i]*d->stride[i];
	}
	return ptr;
}

void *aml_layout_column_aderef(const struct aml_layout_data *data,
			       const size_t *coords)
{
	const struct aml_layout_data_native *d =
	    (const struct aml_layout_data_native *)data;
	void *ptr;
	assert(d != NULL);
	assert(d->ptr != NULL);
	ptr = d->ptr;
	for(size_t i = 0; i < d->ndims; i++)
	{
		assert(coords[i] < d->dims[i]);
		ptr += coords[i]*d->cpitch[i]*d->stride[i];
	}
	return ptr;
}

int aml_layout_column_order(const struct aml_layout_data *data)
{
	return AML_TYPE_LAYOUT_COLUMN_ORDER;
}

int aml_layout_column_dims(const struct aml_layout_data *data, va_list dims)
{
	const struct aml_layout_data_native *d =
	    (const struct aml_layout_data_native *)data;
	assert(d != NULL);
	for(size_t i = 0; i < d->ndims; i++)
	{
		size_t *dim = va_arg(dims, size_t*);
		assert(dim != NULL);
		*dim = d->dims[i];
	}
	return 0;
}

int aml_layout_column_adims(const struct aml_layout_data *data, size_t *dims)
{
	const struct aml_layout_data_native *d =
	    (const struct aml_layout_data_native *)data;
	assert(d != NULL);
	assert(dims != NULL);
	memcpy((void*)dims, (void*)d->dims, sizeof(size_t)*d->ndims);
	return 0;
}

size_t aml_layout_column_ndims(const struct aml_layout_data *data)
{
	const struct aml_layout_data_native *d =
	    (const struct aml_layout_data_native *)data;
	return d->ndims;
}

size_t aml_layout_column_element_size(const struct aml_layout_data *data)
{
	const struct aml_layout_data_native *d =
	    (const struct aml_layout_data_native *)data;
	return d->cpitch[0];
}

struct aml_layout_ops aml_layout_column_ops = {
	aml_layout_column_deref,
	aml_layout_column_aderef,
	aml_layout_column_order,
	aml_layout_column_dims,
	aml_layout_column_adims,
	aml_layout_column_ndims,
	aml_layout_column_element_size
};


/*******************************************************************************
 * ROW OPERATORS:
 ******************************************************************************/

void *aml_layout_row_deref(const struct aml_layout_data *data, va_list coords)
{
	const struct aml_layout_data_native *d =
	    (const struct aml_layout_data_native *)data;
	void *ptr;
	assert(d != NULL);
	assert(d->ptr != NULL);
	ptr = d->ptr;
	for(size_t i = 0; i < d->ndims; i++)
	{
		size_t c = va_arg(coords, size_t);
		assert(c < d->dims[d->ndims - i - 1]);
		ptr += c * d->cpitch[d->ndims - i - 1] *
			   d->stride[d->ndims - i - 1];
	}
	return ptr;
}

void *aml_layout_row_aderef(const struct aml_layout_data *data,
			    const size_t *coords)
{
	const struct aml_layout_data_native *d =
	    (const struct aml_layout_data_native *)data;
	void *ptr;
	assert(d != NULL);
	assert(d->ptr != NULL);
	ptr = d->ptr;
	for(size_t i = 0; i < d->ndims; i++)
	{
		size_t c = coords[i];
		assert(c < d->dims[d->ndims - i - 1]);
		ptr += c * d->cpitch[d->ndims - i - 1] *
			   d->stride[d->ndims - i - 1];
	}
	return ptr;
}

int aml_layout_row_order(const struct aml_layout_data *data)
{
	return AML_TYPE_LAYOUT_ROW_ORDER;
}

int aml_layout_row_dims(const struct aml_layout_data *data, va_list dims)
{
	const struct aml_layout_data_native *d =
	    (const struct aml_layout_data_native *)data;
	assert(d != NULL);
	for(size_t i = 0; i < d->ndims; i++)
	{
		size_t *dim = va_arg(dims, size_t*);
		assert(dim != NULL);
		*dim = d->dims[d->ndims - i - 1];
	}
	return 0;
}

int aml_layout_row_adims(const struct aml_layout_data *data, size_t *dims)
{
	const struct aml_layout_data_native *d =
	    (const struct aml_layout_data_native *)data;
	assert(d != NULL);
	for(size_t i = 0; i < d->ndims; i++)
	{
		dims[i] = d->dims[d->ndims - i - 1];
	}
	return 0;
}

size_t aml_layout_row_ndims(const struct aml_layout_data *data)
{
	const struct aml_layout_data_native *d =
	    (const struct aml_layout_data_native *)data;
	return d->ndims;
}

size_t aml_layout_row_element_size(const struct aml_layout_data *data)
{
	const struct aml_layout_data_native *d =
	    (const struct aml_layout_data_native *)data;
	return d->cpitch[0];
}

struct aml_layout_ops aml_layout_row_ops = {
	aml_layout_row_deref,
	aml_layout_row_aderef,
	aml_layout_row_order,
	aml_layout_row_dims,
	aml_layout_row_adims,
	aml_layout_row_ndims,
	aml_layout_row_element_size
};

