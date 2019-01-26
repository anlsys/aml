#include <aml.h>

int aml_layout_pad_struct_init(struct aml_layout *layout, size_t ndims,
			       size_t element_size, void *memory)
{
	struct aml_layout_data_pad *dataptr;

	assert(layout == (struct aml_layout *)memory);
	memory = (void *)((uintptr_t)memory + sizeof(struct aml_layout));
	dataptr = memory;
	layout->data = memory;
	memory = (void *)((uintptr_t)memory +
		      sizeof(struct aml_layout_data_pad));
	dataptr->target = NULL;
	dataptr->ndims = ndims;
	dataptr->element_size = element_size;
	dataptr->dims = (size_t *)memory;
	dataptr->target_dims = dataptr->dims + ndims;
	dataptr->neutral = (void *)(dataptr->target_dims + ndims);
	return 0;
}

int aml_layout_pad_ainit(struct aml_layout *layout, uint64_t tags,
			 struct aml_layout *target, const size_t *dims,
			 void *neutral)
{
	assert(layout != NULL);
	assert(layout->data != NULL);
	struct aml_layout_data_pad *data =
	    (struct aml_layout_data_pad *)layout->data;
	size_t ndims = aml_layout_ndims(target);
	size_t element_size = aml_layout_element_size(target);
	assert(data->ndims == ndims);
	assert(data->element_size == element_size);
	assert(data->dims);
	assert(data->target_dims);
	assert(data->neutral);
	int type = AML_TYPE_GET(tags, AML_TYPE_LAYOUT_ORDER);
	if (type == AML_TYPE_LAYOUT_ROW_ORDER) {
		AML_TYPE_SET(layout->tags, AML_TYPE_LAYOUT_ORDER,
			     AML_TYPE_LAYOUT_ROW_ORDER);
		layout->ops = &aml_layout_pad_row_ops;
		for(size_t i = 0; i < ndims; i++)
			data->dims[i] = dims[ndims-i-1];
	} else if (type == AML_TYPE_LAYOUT_COLUMN_ORDER) {
		AML_TYPE_SET(layout->tags, AML_TYPE_LAYOUT_ORDER,
			     AML_TYPE_LAYOUT_COLUMN_ORDER);
		layout->ops = &aml_layout_pad_column_ops;
		memcpy(data->dims, dims, ndims * sizeof(size_t));
	}
	type = aml_layout_order(target);
	if(type == AML_TYPE_LAYOUT_ROW_ORDER) {
		size_t target_dims[ndims];
		aml_layout_adims(target, target_dims);
		for(size_t i = 0; i < ndims; i++)
			data->target_dims[i] = target_dims[ndims-i-1];
	} else if (type == AML_TYPE_LAYOUT_COLUMN_ORDER) {
		aml_layout_adims(target, data->target_dims);
	}
	for(size_t i = 0; i < ndims; i++)
		assert(data->dims[i] >= data->target_dims[i]);
	memcpy(data->neutral, neutral, element_size);
	data->target = target;
	return 0;
}

int aml_layout_pad_vinit(struct aml_layout *layout, uint64_t tags,
			 struct aml_layout *target, va_list ap)
{
	size_t ndims = aml_layout_ndims(target);
	size_t dims[ndims];
	for(size_t i = 0; i < ndims; i++)
		dims[i] = va_arg(ap, size_t);
	void *neutral = va_arg(ap, void *);
	return aml_layout_pad_ainit(layout, tags, target, dims, neutral);
}

int aml_layout_pad_init(struct aml_layout *layout, uint64_t tags,
			struct aml_layout *target, ...)
{
	int err;
	va_list ap;
	va_start(ap, target);
	err = aml_layout_pad_vinit(layout, tags, target, ap);
	va_end(ap);
	return err;
}

int aml_layout_pad_acreate(struct aml_layout **layout, uint64_t tags,
			   struct aml_layout *target, const size_t *dims,
			   void *neutral)
{
	assert(target != NULL);
	assert(target->ops != NULL);
	size_t ndims = aml_layout_ndims(target);
	size_t element_size = aml_layout_element_size(target);
	void *baseptr = calloc(1, AML_LAYOUT_PAD_ALLOCSIZE(ndims,
							   element_size));
	*layout = (struct aml_layout *)baseptr;
	aml_layout_pad_struct_init(*layout, ndims, element_size, baseptr);
	return aml_layout_pad_ainit(*layout, tags, target, dims, neutral);
}

int aml_layout_pad_vcreate(struct aml_layout **layout, uint64_t tags,
			   struct aml_layout *target, va_list ap)
{
	assert(target != NULL);
	assert(target->ops != NULL);
	size_t ndims = aml_layout_ndims(target);
	size_t element_size = aml_layout_element_size(target);
	void *baseptr = calloc(1, AML_LAYOUT_PAD_ALLOCSIZE(ndims,
							   element_size));
	*layout = (struct aml_layout *)baseptr;
	aml_layout_pad_struct_init(*layout, ndims, element_size, baseptr);
	return aml_layout_pad_vinit(*layout, tags, target, ap);
}

int aml_layout_pad_create(struct aml_layout **layout, uint64_t tags,
			  struct aml_layout *target, ...)
{
	int err;
	va_list ap;
	assert(target != NULL);
	assert(target->ops != NULL);
	size_t ndims = aml_layout_ndims(target);
	size_t element_size = aml_layout_element_size(target);
	void *baseptr = calloc(1, AML_LAYOUT_PAD_ALLOCSIZE(ndims,
							   element_size));
	*layout = (struct aml_layout *)baseptr;
	aml_layout_pad_struct_init(*layout, ndims, element_size, baseptr);
	va_start(ap, target);
	err = aml_layout_pad_vinit(*layout, tags, target, ap);
	va_end(ap);
	return err;
}

/*******************************************************************************
 * COLUMN OPERATORS:
 ******************************************************************************/

void *aml_layout_pad_column_aderef(const struct aml_layout_data *data,
				   const size_t *coords)
{
	const struct aml_layout_data_pad *d =
	    (const struct aml_layout_data_pad *)data;
	assert(d !=NULL);
	size_t ndims = d->ndims;
	for (int i = 0; i < ndims; i++)
		assert(coords[i] < d->dims[i]);
	for (int i = 0; i < ndims; i++) {
		if(coords[i] >= d->target_dims[i])
			return d->neutral;
	}
	int type = aml_layout_order(d->target);
	if (type == AML_TYPE_LAYOUT_COLUMN_ORDER)
		return aml_layout_aderef(d->target, coords);
	else {
		size_t target_coords[ndims];
		for (int i = 0; i < ndims; i++)
			target_coords[i] = coords[ndims - i - 1];
		return aml_layout_aderef(d->target, coords);
	}
}

void *aml_layout_pad_column_deref(const struct aml_layout_data *data,
				  va_list coords)
{
	const struct aml_layout_data_pad *d =
	    (const struct aml_layout_data_pad *)data;
	assert(d !=NULL);
	size_t ndims = d->ndims;
	size_t target_coords[d->ndims];
	for (int i = 0; i < ndims; i++)
		target_coords[i] = va_arg(coords, size_t);
	return aml_layout_pad_column_aderef(data, target_coords); 
}

int aml_layout_pad_column_order(const struct aml_layout_data *data)
{
	return AML_TYPE_LAYOUT_COLUMN_ORDER;
}

int aml_layout_pad_column_dims(const struct aml_layout_data *data, va_list dims)
{
	const struct aml_layout_data_pad *d =
	    (const struct aml_layout_data_pad *)data;
	assert(d != NULL);
	for(size_t i = 0; i < d->ndims; i++)
	{
		size_t *dim = va_arg(dims, size_t*);
		assert(dim != NULL);
		*dim = d->dims[i];
	}
	return 0;
}

int aml_layout_pad_column_adims(const struct aml_layout_data *data,
				size_t *dims)
{
	const struct aml_layout_data_pad *d =
	    (const struct aml_layout_data_pad *)data;
	assert(d != NULL);
	assert(dims != NULL);
	memcpy((void*)dims, (void*)d->dims, sizeof(size_t)*d->ndims);
	return 0;
}

size_t aml_layout_pad_ndims(const struct aml_layout_data *data)
{
	const struct aml_layout_data_pad *d =
	    (const struct aml_layout_data_pad *)data;
	return d->ndims;
}

size_t aml_layout_pad_element_size(const struct aml_layout_data *data)
{
	const struct aml_layout_data_pad *d =
	    (const struct aml_layout_data_pad *)data;
	return d->element_size;
}

struct aml_layout_ops aml_layout_pad_column_ops = {
	aml_layout_pad_column_deref,
	aml_layout_pad_column_aderef,
	aml_layout_pad_column_order,
	aml_layout_pad_column_dims,
	aml_layout_pad_column_adims,
	aml_layout_pad_column_adims,
	aml_layout_pad_ndims,
	aml_layout_pad_element_size,
	NULL,
	NULL
};

/*******************************************************************************
 * ROW OPERATORS:
 ******************************************************************************/

void *aml_layout_pad_row_aderef(const struct aml_layout_data *data,
				  const size_t *coords)
{
	const struct aml_layout_data_pad *d =
	    (const struct aml_layout_data_pad *)data;
	assert(d !=NULL);
	size_t ndims = d->ndims;
	for (int i = 0; i < ndims; i++)
		assert(coords[ndims - i - 1] < d->dims[i]);
	for (int i = 0; i < ndims; i++) {
		if(coords[ndims - i - 1] >= d->target_dims[i])
			return d->neutral;
	}
	int type = aml_layout_order(d->target);
	if (type == AML_TYPE_LAYOUT_ROW_ORDER)
		return aml_layout_aderef(d->target, coords);
	else {
		size_t target_coords[ndims];
		for (int i = 0; i < ndims; i++)
			target_coords[i] = coords[ndims - i - 1];
		return aml_layout_aderef(d->target, coords);
	}
}

void *aml_layout_pad_row_deref(const struct aml_layout_data *data,
				 va_list coords)
{
	const struct aml_layout_data_pad *d =
	    (const struct aml_layout_data_pad *)data;
	assert(d !=NULL);
	size_t ndims = d->ndims;
	size_t target_coords[d->ndims];
	for (int i = 0; i < ndims; i++)
		target_coords[i] = va_arg(coords, size_t);
	return aml_layout_pad_row_aderef(data, target_coords); 
}

int aml_layout_pad_row_order(const struct aml_layout_data *data)
{
	return AML_TYPE_LAYOUT_ROW_ORDER;
}

int aml_layout_pad_row_dims(const struct aml_layout_data *data, va_list dims)
{
	const struct aml_layout_data_pad *d =
	    (const struct aml_layout_data_pad *)data;
	assert(d != NULL);
	for(size_t i = 0; i < d->ndims; i++)
	{
		size_t *dim = va_arg(dims, size_t*);
		assert(dim != NULL);
		*dim = d->dims[d->ndims - i - 1];
	}
	return 0;
}

int aml_layout_pad_row_adims(const struct aml_layout_data *data, size_t *dims)
{
	const struct aml_layout_data_pad *d =
	    (const struct aml_layout_data_pad *)data;
	assert(d != NULL);
	for(size_t i = 0; i < d->ndims; i++)
	{
		dims[i] = d->dims[d->ndims - i - 1];
	}
	return 0;
}

struct aml_layout_ops aml_layout_pad_row_ops = {
	aml_layout_pad_row_deref,
	aml_layout_pad_row_aderef,
	aml_layout_pad_row_order,
	aml_layout_pad_row_dims,
	aml_layout_pad_row_adims,
	aml_layout_pad_column_adims,
	aml_layout_pad_ndims,
	aml_layout_pad_element_size,
	NULL,
	NULL
};

