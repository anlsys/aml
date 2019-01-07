#include <aml.h>

/*******************************************************************************
 * COLUMN OPERATORS:
 ******************************************************************************/

void *aml_layout_column_deref(const struct aml_layout_data *d, va_list coords)
{
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

void *aml_layout_column_aderef(const struct aml_layout_data *d, const size_t *coords)
{
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

int aml_layout_column_order(const struct aml_layout_data *d)
{
	return AML_TYPE_LAYOUT_COLUMN_ORDER;
}

int aml_layout_column_dims(const struct aml_layout_data *d, va_list dims)
{
	assert(d != NULL);
	for(size_t i = 0; i < d->ndims; i++)
	{
		size_t *dim = va_arg(dims, size_t*);
		assert(dim != NULL);
		*dim = d->dims[i];
	}
	return 0;
}

int aml_layout_column_adims(const struct aml_layout_data *d, size_t *dims)
{
	assert(d != NULL);
	assert(dims != NULL);
	memcpy((void*)dims, (void*)d->dims, sizeof(size_t)*d->ndims);
	return 0;
}

size_t aml_layout_column_ndims(const struct aml_layout_data *d)
{
	return d->ndims;
}

size_t aml_layout_column_element_size(const struct aml_layout_data *d)
{
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

void *aml_layout_row_deref(const struct aml_layout_data *d, va_list coords)
{
	void *ptr;
	assert(d != NULL);
	assert(d->ptr != NULL);
	ptr = d->ptr;
	for(size_t i = 0; i < d->ndims; i++)
	{
		size_t c = va_arg(coords, size_t);
		assert(c < d->dims[d->ndims - i - 1]);
		ptr += c*d->cpitch[d->ndims - i - 1]*d->stride[d->ndims - i - 1];
	}
	return ptr;
}

void *aml_layout_row_aderef(const struct aml_layout_data *d, const size_t *coords)
{
	void *ptr;
	assert(d != NULL);
	assert(d->ptr != NULL);
	ptr = d->ptr;
	for(size_t i = 0; i < d->ndims; i++)
	{
		size_t c = coords[i];
		assert(c < d->dims[d->ndims - i - 1]);
		ptr += c*d->cpitch[d->ndims - i - 1]*d->stride[d->ndims - i - 1];
	}
	return ptr;
}

int aml_layout_row_order(const struct aml_layout_data *d)
{
	return AML_TYPE_LAYOUT_ROW_ORDER;
}

int aml_layout_row_dims(const struct aml_layout_data *d, va_list dims)
{
	assert(d != NULL);
	for(size_t i = 0; i < d->ndims; i++)
	{
		size_t *dim = va_arg(dims, size_t*);
		assert(dim != NULL);
		*dim = d->dims[d->ndims - i - 1];
	}
	return 0;
}

int aml_layout_row_adims(const struct aml_layout_data *d, size_t *dims)
{
	assert(d != NULL);
	for(size_t i = 0; i < d->ndims; i++)
	{
		dims[i] = d->dims[d->ndims - i - 1];
	}
	return 0;
}

size_t aml_layout_row_ndims(const struct aml_layout_data *d)
{
	return d->ndims;
}

size_t aml_layout_row_element_size(const struct aml_layout_data *d)
{
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

