#include <aml.h>

/*******************************************************************************
 * General API: common operators:
 ******************************************************************************/

void *aml_layout_deref(const struct aml_layout *layout, ...)
{
	assert(layout != NULL);
	assert(layout->ops != NULL);
	va_list ap;
	void *ret;
	va_start(ap, layout);
	ret = layout->ops->deref(layout->data, ap);
	va_end(ap);
	return ret;
}

int aml_layout_order(const struct aml_layout *layout)
{
	assert(layout != NULL);
	assert(layout->ops != NULL);
	return layout->ops->order(layout->data);
}

int aml_layout_dims(const struct aml_layout *layout, ...)
{
	assert(layout != NULL);
	assert(layout->ops != NULL);
	va_list ap;
	int ret;
	va_start(ap, layout);
	ret = layout->ops->dims(layout->data, ap);
	va_end(ap);
	return ret;
}

/*******************************************************************************
 * Layout initialization:
 ******************************************************************************/

int aml_layout_struct_init(struct aml_layout *layout,
			   size_t ndims, void *memory)
{
	assert(layout == (struct aml_layout *)memory);
	memory = (void *)((uintptr_t)memory +
		      sizeof(struct aml_layout));
	layout->data = memory;
	memory = (void *)((uintptr_t)memory +
		      sizeof(struct aml_layout_data));
	layout->data->ndims = ndims;
	layout->data->dims = (size_t *)memory;
	layout->data->pitch = layout->data->dims + ndims;
	layout->data->stride = layout->data->pitch + ndims;
	return 0;
}

int aml_layout_ainit(struct aml_layout *layout, uint64_t tags, void *ptr,
		     const size_t element_size, size_t ndims,
		     const size_t *dims, const size_t *stride,
		     const size_t *pitch)
{
	assert(layout != NULL);
	assert(layout->data != NULL);
	struct aml_layout_data *data = layout->data;
	assert(data->ndims == ndims);
	assert(data->dims);
	assert(data->pitch);
	assert(data->stride);
	data->ptr = ptr;
	if(tags == AML_TYPE_LAYOUT_COLUMN_ORDER)
	{
		layout->tags = tags;
		layout->ops = &aml_layout_column_ops;
		for(size_t i = 0; i < ndims; i++)
		{
			data->dims[i] = dims[ndims-i-1];
			data->stride[i] = stride[ndims-i-1];
		}
		data->pitch[0] = element_size;
		for(size_t i = 1; i < ndims; i++)
			data->pitch[i] = data->pitch[i-1]*pitch[ndims-i-1];
	}
	else if(tags == AML_TYPE_LAYOUT_ROW_ORDER)
	{
		layout->tags = tags;
		layout->ops = &aml_layout_row_ops;
		memcpy(data->dims, dims, ndims * sizeof(size_t));
		/* pitches are only necessary for ndims-1 dimensions. Since we
		 * store element size as p->pitch[0], there's still ndims
		 * elements in the array.
		 */
		data->pitch[0] = element_size;
		for(size_t i = 1; i < ndims; i++)
			data->pitch[i] = data->pitch[i-1]*pitch[i-1];
		memcpy(data->stride, stride, ndims * sizeof(size_t));
	}
	return 0;
}

int aml_layout_vinit(struct aml_layout *p, uint64_t tags, void *ptr,
		     const size_t element_size, size_t ndims, va_list ap)
{
	size_t dims[ndims];
	size_t stride[ndims];
	size_t pitch[ndims-1];
	for(size_t i = 0; i < ndims; i++)
		dims[i] = va_arg(ap, size_t);
	for(size_t i = 0; i < ndims; i++)
		stride[i] = va_arg(ap, size_t);
	for(size_t i = 0; i < ndims-1; i++)
		pitch[i] = va_arg(ap, size_t);
	return aml_layout_ainit(p, tags, ptr, element_size, ndims, dims, stride,
			       pitch);
}

int aml_layout_init(struct aml_layout *p, uint64_t tags, void *ptr,
		     const size_t element_size, size_t ndims, ...)
{
	int err;
	va_list ap;
	va_start(ap, ndims);
	err = aml_layout_vinit(p, tags, ptr, element_size, ndims, ap);
	va_end(ap);
	return err;
}

int aml_layout_acreate(struct aml_layout **layout, uint64_t tags, void *ptr,
		       const size_t element_size,
		       size_t ndims, const size_t *dims, const size_t *stride,
		       const size_t *pitch)
{
	assert(ndims > 0);
	void *baseptr = calloc(1, AML_LAYOUT_ALLOCSIZE(ndims));
	*layout = (struct aml_layout *)baseptr;
	aml_layout_struct_init(*layout, ndims, baseptr);
	aml_layout_init(*layout, tags, ptr, element_size, ndims, dims, stride, pitch);
	return 0;
}

int aml_layout_vcreate(struct aml_layout **layout, uint64_t tags, void *ptr,
		       const size_t element_size, size_t ndims, va_list ap)
{
	assert(ndims > 0);
	void *baseptr = calloc(1, AML_LAYOUT_ALLOCSIZE(ndims));
	*layout = (struct aml_layout *)baseptr;
	aml_layout_struct_init(*layout, ndims, baseptr);
	return aml_layout_vinit(*layout, tags, ptr, element_size, ndims, ap);
}

int aml_layout_create(struct aml_layout **layout, uint64_t tags, void *ptr,
		      const size_t element_size, size_t ndims, ...)
{
	int err;
	va_list ap;
	assert(ndims > 0);
	void *baseptr = calloc(1, AML_LAYOUT_ALLOCSIZE(ndims));
	*layout = (struct aml_layout *)baseptr;
	aml_layout_struct_init(*layout, ndims, baseptr);
	va_start(ap, ndims);
	err = aml_layout_vinit(*layout, tags, ptr, element_size, ndims, ap);
	va_end(ap);
	return err;
}
