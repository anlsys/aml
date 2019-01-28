#include <aml.h>

int aml_tiling_nd_resize_struct_init(struct aml_tiling_nd *t, size_t ndims,
				     void *memory)
{
	struct aml_tiling_nd_data_resize *dataptr;

	assert(t == (struct aml_tiling_nd *)memory);
	memory = (void *)((uintptr_t)memory +
		     sizeof(struct aml_tiling_nd));
	dataptr = memory;
	t->data = memory;
	memory = (void *)((uintptr_t)memory +
		     sizeof(struct aml_tiling_nd_data_resize));
	dataptr->l = NULL;
	dataptr->ndims = ndims;
	dataptr->tile_dims = (size_t *)memory;
	dataptr->dims = dataptr->tile_dims + ndims;
	dataptr->border_tile_dims = dataptr->dims + ndims;
	return 0;
}

int aml_tiling_nd_resize_ainit(struct aml_tiling_nd *t, uint64_t tags,
                               const struct aml_layout *l, size_t ndims,
                               const size_t *tile_dims)
{
	assert(t != NULL);
	assert(t->data != NULL);
	struct aml_tiling_nd_data_resize *data =
	    (struct aml_tiling_nd_data_resize *)t->data;
	assert(data->ndims == ndims);
	assert(data->tile_dims);
	assert(data->dims);
	assert(data->border_tile_dims);
	data->l = l;
	int type = AML_TYPE_GET(tags, AML_TYPE_TILING_ORDER);
	if (type == AML_TYPE_TILING_ROW_ORDER) {
		AML_TYPE_SET(t->tags, AML_TYPE_TILING_ORDER,
			     AML_TYPE_TILING_ROW_ORDER);
		t->ops = &aml_tiling_nd_resize_row_ops;
		for (size_t i = 0; i < ndims; i++)
			data->tile_dims[i] = tile_dims[ndims-i-1];
	} else {
		AML_TYPE_SET(t->tags, AML_TYPE_TILING_ORDER,
			     AML_TYPE_TILING_COLUMN_ORDER);
		t->ops = &aml_tiling_nd_resize_column_ops;
		for (size_t i = 0; i < ndims; i++)
			data->tile_dims[i] = tile_dims[i];
	}
	size_t target_dims[ndims];
	aml_layout_adims_column(l, target_dims);
	for (size_t i = 0; i < ndims; i++) {
		data->border_tile_dims[i] = target_dims[i] % data->tile_dims[i];
		data->dims[i] = target_dims[i] / data->tile_dims[i];
		if (data->border_tile_dims[i] == 0)
			data->border_tile_dims[i] = target_dims[i];
		else
			data->dims[i] += 1;
	}
	return 0;
}

int aml_tiling_nd_resize_vinit(struct aml_tiling_nd *t, uint64_t tags,
                               const struct aml_layout *l, size_t ndims,
                               va_list data)
{
	size_t tile_dims[ndims];
	for(size_t i = 0; i < ndims; i++)
		tile_dims[i] = va_arg(data, size_t);
	return aml_tiling_nd_resize_ainit(t, tags, l, ndims, tile_dims);
}

int aml_tiling_nd_resize_init(struct aml_tiling_nd *t, uint64_t tags,
			      const struct aml_layout *l, size_t ndims, ...)
{
	int err;
	va_list ap;
	va_start(ap, ndims);
	err = aml_tiling_nd_resize_vinit(t, tags, l, ndims, ap);
	va_end(ap);
	return err;
}

int aml_tiling_nd_resize_acreate(struct aml_tiling_nd **t, uint64_t tags,
				 const struct aml_layout *l, size_t ndims,
				 const size_t *tile_dims)
{
	assert(ndims > 0);
	void *baseptr = calloc(1, AML_TILING_RESIZE_ALLOCSIZE(ndims));
	*t = (struct aml_tiling_nd *)baseptr;
	aml_tiling_nd_resize_struct_init(*t, ndims, baseptr);
	return aml_tiling_nd_resize_ainit(*t, tags, l, ndims, tile_dims);
}

int aml_tiling_nd_resize_vcreate(struct aml_tiling_nd **t, uint64_t tags,
				 const struct aml_layout *l, size_t ndims,
				 va_list data)
{
	assert(ndims > 0);
	void *baseptr = calloc(1, AML_TILING_RESIZE_ALLOCSIZE(ndims));
	*t = (struct aml_tiling_nd *)baseptr;
	aml_tiling_nd_resize_struct_init(*t, ndims, baseptr);
	return aml_tiling_nd_resize_vinit(*t, tags, l, ndims, data);
}

int aml_tiling_nd_resize_create(struct aml_tiling_nd **t, uint64_t tags,
				const struct aml_layout *l, size_t ndims, ...)
{
	int err;
	va_list ap;
	assert(ndims > 0);
	void *baseptr = calloc(1, AML_TILING_RESIZE_ALLOCSIZE(ndims));
	*t = (struct aml_tiling_nd *)baseptr;
	aml_tiling_nd_resize_struct_init(*t, ndims, baseptr);
	va_start(ap, ndims);
	err = aml_tiling_nd_resize_vinit(*t, tags, l, ndims, ap);
	va_end(ap);
	return err;
}


