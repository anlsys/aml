#include <aml.h>

int aml_layout_reshape_struct_init(struct aml_layout *layout, size_t ndims,
				   void *memory)
{
	struct aml_layout_data_reshape *dataptr;

	assert(layout == (struct aml_layout *)memory);
	memory = (void *)((uintptr_t)memory + sizeof(struct aml_layout));
        dataptr = memory;
	layout->data = memory;
	memory = (void *)((uintptr_t)memory +
		     sizeof(struct aml_layout_data_reshape));
	dataptr->target = NULL;
	dataptr->ndims = ndims;
	dataptr->dims = (size_t *)memory;
	dataptr->coffsets = dataptr->dims + ndims;
	dataptr->target_dims = dataptr->dims + 2 * ndims;
	return 0;
}

int aml_layout_reshape_ainit(struct aml_layout *layout, uint64_t tags,
			     struct aml_layout *target, size_t ndims,
			     const size_t *dims)
{
	assert(layout != NULL);
	assert(layout->data != NULL);
	struct aml_layout_data_reshape *data =
	    (struct aml_layout_data_reshape *)layout->data;
	size_t target_ndims = aml_layout_ndims(target);
	assert(ndims != 0);
	assert(data->ndims == ndims);
        assert(data->dims);
        assert(data->coffsets);
	assert(data->target_dims);
	data->target_ndims = target_ndims;
	data->target = target;
	assert(data->target_ndims != 0);
	int type = AML_TYPE_GET(tags, AML_TYPE_LAYOUT_ORDER);
	if (type == AML_TYPE_LAYOUT_ROW_ORDER) {
		AML_TYPE_SET(layout->tags, AML_TYPE_LAYOUT_ORDER,
			     AML_TYPE_LAYOUT_ROW_ORDER);
		layout->ops = &aml_layout_reshape_row_ops;
		for(size_t i = 0; i < ndims; i++)
			data->dims[i] = dims[ndims-i-1];
	} else {
		AML_TYPE_SET(layout->tags, AML_TYPE_LAYOUT_ORDER,
			     AML_TYPE_LAYOUT_COLUMN_ORDER);
		layout->ops = &aml_layout_reshape_column_ops;
		memcpy(data->dims, dims, ndims * sizeof(size_t));
	}
	type = aml_layout_order(target);
	if(type == AML_TYPE_LAYOUT_ROW_ORDER) {
		size_t target_dims[target_ndims];
		aml_layout_adims(target, target_dims);
		for(size_t i = 0; i < target_ndims; i++)
			data->target_dims[i] = target_dims[target_ndims-i-1];
	} else {
		aml_layout_adims(target, data->target_dims);
	}
	size_t prod, target_prod;
	prod = 1;
	for(size_t i = 0; i < ndims; i++) {
		data->coffsets[i] = prod;
		prod *= data->dims[i];
	}
	target_prod = 1;
	for(size_t i = 0; i < data->target_ndims; i++)
		target_prod *= data->target_dims[i];
	assert(target_prod == prod);
	return 0;
}

int aml_layout_reshape_vinit(struct aml_layout *layout, uint64_t tags,
			     struct aml_layout *target, size_t ndims,
			     va_list data)
{
	size_t dims[ndims];
	for(size_t i = 0; i < ndims; i++)
		dims[i] = va_arg(data, size_t);
	return aml_layout_reshape_ainit(layout, tags, target, ndims, dims);
}

int aml_layout_reshape_init(struct aml_layout *layout, uint64_t tags,
			    struct aml_layout *target, size_t ndims, ...)
{
	int err;
	va_list ap;
	va_start(ap, ndims);
	err = aml_layout_reshape_vinit(layout, tags, target, ndims, ap);
	va_end(ap);
	return err;
}

int aml_layout_reshape_acreate(struct aml_layout **layout, uint64_t tags,
			       struct aml_layout *target, size_t ndims,
			       const size_t *dims)
{
	assert(target != NULL);
	assert(target->ops != NULL);
	size_t target_ndims = aml_layout_ndims(target);
	void *baseptr = calloc(1, AML_LAYOUT_RESHAPE_ALLOCSIZE(ndims,
							       target_ndims));
	assert(baseptr != NULL);
	*layout = (struct aml_layout *)baseptr;
	aml_layout_reshape_struct_init(*layout, ndims, baseptr);
	return aml_layout_reshape_ainit(*layout, tags, target, ndims, dims);
}

int aml_layout_reshape_vcreate(struct aml_layout **layout, uint64_t tags,
			       struct aml_layout *target, size_t ndims,
			       va_list data)
{
	assert(target != NULL);
	assert(target->ops != NULL);
	size_t target_ndims = aml_layout_ndims(target);
	void *baseptr = calloc(1, AML_LAYOUT_RESHAPE_ALLOCSIZE(ndims,
							       target_ndims));
	assert(baseptr != NULL);
	*layout = (struct aml_layout *)baseptr;
	aml_layout_reshape_struct_init(*layout, ndims, baseptr);
	return aml_layout_reshape_vinit(*layout, tags, target, ndims, data);
}

int aml_layout_reshape_create(struct aml_layout **layout, uint64_t tags,
			      struct aml_layout *target, size_t ndims, ...)
{
	int err;
	va_list data;
	assert(target != NULL);
	assert(target->ops != NULL);
	size_t target_ndims = aml_layout_ndims(target);
	void *baseptr = calloc(1, AML_LAYOUT_RESHAPE_ALLOCSIZE(ndims,
							       target_ndims));
	assert(baseptr != NULL);
	*layout = (struct aml_layout *)baseptr;
	aml_layout_reshape_struct_init(*layout, ndims, baseptr);
	va_start(data, ndims);
	err = aml_layout_reshape_vinit(*layout, tags, target, ndims, data);
	va_end(data);
	return err;
}

/*******************************************************************************
 * COLUMN OPERATORS:
 ******************************************************************************/

void *aml_layout_reshape_column_aderef(const struct aml_layout_data *data,
				       const size_t *coords)
{
	const struct aml_layout_data_reshape *d =
	    (const struct aml_layout_data_reshape *)data;
	assert(d !=NULL);

	size_t ndims = d->ndims;

	for (int i = 0; i < ndims; i++)
		assert(coords[i] < d->dims[i]);

	size_t target_ndims = d->target_ndims;
	size_t offset = 0;
	size_t remainder;
	size_t target_coords[target_ndims];

	for (int i = 0; i < ndims; i++)
		offset += coords[i] * d->coffsets[i];

	int type = aml_layout_order(d->target);
	if (type == AML_TYPE_LAYOUT_COLUMN_ORDER) {
		for (int i = 0; i < target_ndims; i++) {
			target_coords[i] = offset % d->target_dims[i];
			offset /= d->target_dims[i];
		}
	} else {
		for (int i = 0; i < target_ndims; i++) {
			target_coords[target_ndims - i - 1] =
			    offset % d->target_dims[i];
			offset /= d->target_dims[i];
		}
	}
	return aml_layout_aderef(d->target, target_coords);
}

void *aml_layout_reshape_column_deref(const struct aml_layout_data *data,
				      va_list coords)
{
	const struct aml_layout_data_reshape *d =
	    (const struct aml_layout_data_reshape *)data;
	assert(d !=NULL);
	size_t target_coords[d->ndims];
	for (int i = 0; i < d->ndims; i++)
		target_coords[i] = va_arg(coords, size_t);
	return aml_layout_reshape_column_aderef(data, target_coords);
}

int aml_layout_reshape_column_order(const struct aml_layout_data *data)
{
	return AML_TYPE_LAYOUT_COLUMN_ORDER;
}

int aml_layout_reshape_column_dims(const struct aml_layout_data *data, va_list dims)
{
	const struct aml_layout_data_reshape *d =
	    (const struct aml_layout_data_reshape *)data;
	assert(d != NULL);
	for(size_t i = 0; i < d->ndims; i++)
	{
		size_t *dim = va_arg(dims, size_t*);
		assert(dim != NULL);
		*dim = d->dims[i];
	}
	return 0;
}

int aml_layout_reshape_column_adims(const struct aml_layout_data *data,
				size_t *dims)
{
	const struct aml_layout_data_reshape *d =
	    (const struct aml_layout_data_reshape *)data;
	assert(d != NULL);
	assert(dims != NULL);
	memcpy((void*)dims, (void*)d->dims, sizeof(size_t)*d->ndims);
	return 0;
}

size_t aml_layout_reshape_ndims(const struct aml_layout_data *data)
{
	const struct aml_layout_data_reshape *d =
	    (const struct aml_layout_data_reshape *)data;
	return d->ndims;
}

size_t aml_layout_reshape_element_size(const struct aml_layout_data *data)
{
	const struct aml_layout_data_reshape *d =
	    (const struct aml_layout_data_reshape *)data;
	return aml_layout_element_size(d->target);
}

struct aml_layout_ops aml_layout_reshape_column_ops = {
	aml_layout_reshape_column_deref,
	aml_layout_reshape_column_aderef,
	aml_layout_reshape_column_order,
	aml_layout_reshape_column_dims,
	aml_layout_reshape_column_adims,
	aml_layout_reshape_column_adims,
	aml_layout_reshape_ndims,
	aml_layout_reshape_element_size,
	NULL,
	NULL
};

/*******************************************************************************
 * ROW OPERATORS:
 ******************************************************************************/

void *aml_layout_reshape_row_aderef(const struct aml_layout_data *data,
				    const size_t *coords)
{
	const struct aml_layout_data_reshape *d =
	    (const struct aml_layout_data_reshape *)data;
	assert(d !=NULL);

	size_t ndims = d->ndims;

	for (int i = 0; i < ndims; i++)
		assert(coords[ndims - i - 1] < d->dims[i]);

	size_t target_ndims = d->target_ndims;
	size_t offset = 0;
	size_t remainder;
	size_t target_coords[target_ndims];

	for (int i = 0; i < ndims; i++)
		offset += coords[ndims - i - 1] * d->coffsets[i];

	int type = aml_layout_order(d->target);
	if (type == AML_TYPE_LAYOUT_COLUMN_ORDER) {
		for (int i = 0; i < target_ndims; i++) {
			target_coords[i] = offset % d->target_dims[i];
			offset /= d->target_dims[i];
		}
	} else {
		for (int i = 0; i < target_ndims; i++) {
			target_coords[target_ndims - i - 1] =
			    offset % d->target_dims[i];
			offset /= d->target_dims[i];
		}
	}
	return aml_layout_aderef(d->target, target_coords);
}

void *aml_layout_reshape_row_deref(const struct aml_layout_data *data,
				   va_list coords)
{
	const struct aml_layout_data_reshape *d =
	    (const struct aml_layout_data_reshape *)data;
	assert(d !=NULL);
	size_t target_coords[d->ndims];
	for (int i = 0; i < d->ndims; i++)
		target_coords[i] = va_arg(coords, size_t);
	return aml_layout_reshape_row_aderef(data, target_coords);
}

int aml_layout_reshape_row_order(const struct aml_layout_data *data)
{
	return AML_TYPE_LAYOUT_ROW_ORDER;
}

int aml_layout_reshape_row_dims(const struct aml_layout_data *data,
				va_list dims)
{
	const struct aml_layout_data_reshape *d =
	    (const struct aml_layout_data_reshape *)data;
	assert(d != NULL);
	for(size_t i = 0; i < d->ndims; i++)
	{
		size_t *dim = va_arg(dims, size_t*);
		assert(dim != NULL);
		*dim = d->dims[d->ndims - i - 1];
	}
	return 0;
}

int aml_layout_reshape_row_adims(const struct aml_layout_data *data,
				 size_t *dims)
{
	const struct aml_layout_data_reshape *d =
	    (const struct aml_layout_data_reshape *)data;
	assert(d != NULL);
	for(size_t i = 0; i < d->ndims; i++)
	{
		dims[i] = d->dims[d->ndims - i - 1];
	}
	return 0;
}

struct aml_layout_ops aml_layout_reshape_row_ops = {
	aml_layout_reshape_row_deref,
	aml_layout_reshape_row_aderef,
	aml_layout_reshape_row_order,
	aml_layout_reshape_row_dims,
	aml_layout_reshape_row_adims,
	aml_layout_reshape_column_adims,
	aml_layout_reshape_ndims,
	aml_layout_reshape_element_size,
	NULL,
	NULL
};


