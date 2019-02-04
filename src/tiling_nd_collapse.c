#include <aml.h>

int aml_tiling_nd_collapse_struct_init(struct aml_tiling_nd *t, size_t ndims,
				     void *memory)
{
	struct aml_tiling_nd_data_collapse *dataptr;

	assert(t == (struct aml_tiling_nd *)memory);
	memory = (void *)((uintptr_t)memory +
		     sizeof(struct aml_tiling_nd));
	dataptr = memory;
	t->data = memory;
	memory = (void *)((uintptr_t)memory +
		     sizeof(struct aml_tiling_nd_data_collapse));
	dataptr->l = NULL;
	dataptr->ndims = ndims;
	dataptr->tile_dims = (size_t *)memory;
	dataptr->dims = dataptr->tile_dims + ndims;
	dataptr->border_tile_dims = dataptr->dims + ndims;
	return 0;
}

int aml_tiling_nd_collapse_ainit(struct aml_tiling_nd *t, uint64_t tags,
                               const struct aml_layout *l, size_t ndims,
                               const size_t *tile_dims)
{
	assert(t != NULL);
	assert(t->data != NULL);
	struct aml_tiling_nd_data_collapse *data =
	    (struct aml_tiling_nd_data_collapse *)t->data;
	assert(data->ndims == ndims);
	assert(data->tile_dims);
	assert(data->dims);
	assert(data->border_tile_dims);
	data->l = l;
	int type = AML_TYPE_GET(tags, AML_TYPE_TILING_ORDER);
	if (type == AML_TYPE_TILING_ROW_ORDER) {
		AML_TYPE_SET(t->tags, AML_TYPE_TILING_ORDER,
			     AML_TYPE_TILING_ROW_ORDER);
		t->ops = &aml_tiling_nd_collapse_row_ops;
		for (size_t i = 0; i < ndims; i++)
			data->tile_dims[i] = tile_dims[ndims-i-1];
	} else {
		AML_TYPE_SET(t->tags, AML_TYPE_TILING_ORDER,
			     AML_TYPE_TILING_COLUMN_ORDER);
		t->ops = &aml_tiling_nd_collapse_column_ops;
		for (size_t i = 0; i < ndims; i++)
			data->tile_dims[i] = tile_dims[i];
	}
	size_t target_dims[ndims];
	l->ops->adims_column(l->data, target_dims);
	for (size_t i = 0; i < ndims; i++) {
		data->border_tile_dims[i] = target_dims[i] % data->tile_dims[i];
		data->dims[i] = target_dims[i] / data->tile_dims[i];
		if (data->border_tile_dims[i] == 0)
			data->border_tile_dims[i] = data->tile_dims[i];
		else
			data->dims[i] += 1;
	}
	return 0;
}

int aml_tiling_nd_collapse_vinit(struct aml_tiling_nd *t, uint64_t tags,
                               const struct aml_layout *l, size_t ndims,
                               va_list data)
{
	size_t tile_dims[ndims];
	for(size_t i = 0; i < ndims; i++)
		tile_dims[i] = va_arg(data, size_t);
	return aml_tiling_nd_collapse_ainit(t, tags, l, ndims, tile_dims);
}

int aml_tiling_nd_collapse_init(struct aml_tiling_nd *t, uint64_t tags,
			      const struct aml_layout *l, size_t ndims, ...)
{
	int err;
	va_list ap;
	va_start(ap, ndims);
	err = aml_tiling_nd_collapse_vinit(t, tags, l, ndims, ap);
	va_end(ap);
	return err;
}

int aml_tiling_nd_collapse_acreate(struct aml_tiling_nd **t, uint64_t tags,
				 const struct aml_layout *l, size_t ndims,
				 const size_t *tile_dims)
{
	assert(ndims > 0);
	void *baseptr = calloc(1, AML_TILING_COLLAPSE_ALLOCSIZE(ndims));
	*t = (struct aml_tiling_nd *)baseptr;
	aml_tiling_nd_collapse_struct_init(*t, ndims, baseptr);
	return aml_tiling_nd_collapse_ainit(*t, tags, l, ndims, tile_dims);
}

int aml_tiling_nd_collapse_vcreate(struct aml_tiling_nd **t, uint64_t tags,
				 const struct aml_layout *l, size_t ndims,
				 va_list data)
{
	assert(ndims > 0);
	void *baseptr = calloc(1, AML_TILING_COLLAPSE_ALLOCSIZE(ndims));
	*t = (struct aml_tiling_nd *)baseptr;
	aml_tiling_nd_collapse_struct_init(*t, ndims, baseptr);
	return aml_tiling_nd_collapse_vinit(*t, tags, l, ndims, data);
}

int aml_tiling_nd_collapse_create(struct aml_tiling_nd **t, uint64_t tags,
				const struct aml_layout *l, size_t ndims, ...)
{
	int err;
	va_list ap;
	assert(ndims > 0);
	void *baseptr = calloc(1, AML_TILING_COLLAPSE_ALLOCSIZE(ndims));
	*t = (struct aml_tiling_nd *)baseptr;
	aml_tiling_nd_collapse_struct_init(*t, ndims, baseptr);
	va_start(ap, ndims);
	err = aml_tiling_nd_collapse_vinit(*t, tags, l, ndims, ap);
	va_end(ap);
	return err;
}

/*----------------------------------------------------------------------------*/

struct aml_layout*
aml_tiling_nd_collapse_column_aindex(const struct aml_tiling_nd_data *l,
				   const size_t *coords)
{
	const struct aml_tiling_nd_data_collapse *d =
	    (const struct aml_tiling_nd_data_collapse *)l;
	assert(d != NULL);
	size_t ndims = d->ndims;
	size_t new_coords[ndims];
	size_t offsets[ndims];
	size_t dims[ndims];
	size_t strides[ndims];
	for(size_t i = 0, j = 0; i < ndims; i++)
		if (d->dims[i] > 1) {
			assert(coords[j] < d->dims[i]);
			new_coords[i] = coords[j];
			j++;
		} else
			new_coords[i] = 0;
	for(size_t i = 0; i < ndims; i++) {
		offsets[i] = new_coords[i] * d->tile_dims[i];
		strides[i] = 1;
	}
	for(size_t i = 0; i < ndims; i++)
		dims[i] = (new_coords[i] == d->dims[i] - 1 ?
			      d->border_tile_dims[i] :
			      d->tile_dims[i] );
	return d->l->ops->aslice_column(d->l->data, offsets, dims, strides);
}

struct aml_layout*
aml_tiling_nd_collapse_column_index(const struct aml_tiling_nd_data *l,
				  va_list coords)
{
	const struct aml_tiling_nd_data_collapse *d =
	    (const struct aml_tiling_nd_data_collapse *)l;
	size_t n_coords[d->ndims];
	for(size_t i = 0, j = 0; i < d->ndims; i++)
		if (d->dims[i] > 1)
			n_coords[j++] = va_arg(coords, size_t);
	return aml_tiling_nd_collapse_column_aindex(l, n_coords);
}

int
aml_tiling_nd_collapse_column_order(const struct aml_tiling_nd_data * l)
{
	return AML_TYPE_TILING_COLUMN_ORDER;
}

int
aml_tiling_nd_collapse_column_tile_dims(const struct aml_tiling_nd_data *l,
				      va_list dims_ptrs)
{
	const struct aml_tiling_nd_data_collapse *d =
	    (const struct aml_tiling_nd_data_collapse *)l;
	assert(d != NULL);
	for(size_t i = 0; i < d->ndims; i++) {
		size_t *dim = va_arg(dims_ptrs, size_t*);
		assert(dim != NULL);
		*dim = d->tile_dims[i];
	}
	return 0;
}

int
aml_tiling_nd_collapse_column_tile_adims(const struct aml_tiling_nd_data *l,
				       size_t *tile_dims)
{
	const struct aml_tiling_nd_data_collapse *d =
	    (const struct aml_tiling_nd_data_collapse *)l;
	assert(d != NULL);
	memcpy((void*)tile_dims, (void*)d->tile_dims, sizeof(size_t)*d->ndims);
	return 0;	
}

int
aml_tiling_nd_collapse_column_dims(const struct aml_tiling_nd_data *l,
				 va_list dims_ptrs)
{
	const struct aml_tiling_nd_data_collapse *d =
	    (const struct aml_tiling_nd_data_collapse *)l;
	assert(d != NULL);
	for(size_t i = 0; i < d->ndims; i++) {
		if (d->dims[i] > 1) {
			size_t *dim = va_arg(dims_ptrs, size_t*);
			assert(dim != NULL);
			*dim = d->dims[i];
		}
	}
	return 0;
}

int
aml_tiling_nd_collapse_column_adims(const struct aml_tiling_nd_data *l,
				  size_t *dims)
{
	const struct aml_tiling_nd_data_collapse *d =
	    (const struct aml_tiling_nd_data_collapse *)l;
	assert(d != NULL);
	for(size_t i = 0, j = 0; i < d->ndims; i++)
		if (d->dims[i] > 1)
			dims[j++] = d->dims[i];
	return 0;	
}

size_t
aml_tiling_nd_collapse_column_ndims(const struct aml_tiling_nd_data *l)
{
	const struct aml_tiling_nd_data_collapse *d =
	    (const struct aml_tiling_nd_data_collapse *)l;
	assert(d != NULL);
	size_t ndims = 0;
	for(size_t i = 0; i < d->ndims; i++)
		if (d->dims[i] > 1)
			ndims++;
	return ndims;
}

struct aml_tiling_nd_ops aml_tiling_nd_collapse_column_ops = {
	aml_tiling_nd_collapse_column_index,
	aml_tiling_nd_collapse_column_aindex,
	aml_tiling_nd_collapse_column_order,
	aml_tiling_nd_collapse_column_tile_dims,
	aml_tiling_nd_collapse_column_tile_adims,
	aml_tiling_nd_collapse_column_dims,
	aml_tiling_nd_collapse_column_adims,
	aml_tiling_nd_collapse_column_ndims
};

/*----------------------------------------------------------------------------*/

struct aml_layout*
aml_tiling_nd_collapse_row_aindex(const struct aml_tiling_nd_data *l,
				   const size_t *coords)
{
	const struct aml_tiling_nd_data_collapse *d =
	    (const struct aml_tiling_nd_data_collapse *)l;
	assert(d != NULL);
	size_t ndims = d->ndims;
	size_t new_coords[ndims];
	size_t offsets[ndims];
	size_t dims[ndims];
	size_t strides[ndims];

	for(size_t i = 0, j = 0; i < ndims; i++)
		if (d->dims[ndims - i - 1] > 1) {
			assert(coords[j] < d->dims[ndims - i - 1]);
			new_coords[ndims - i - 1] = coords[j];
			j++;
		} else
			new_coords[ndims - i - 1] = 0;
	for(size_t i = 0; i < ndims; i++) {
		
		offsets[i] = new_coords[i] * d->tile_dims[i];
		strides[i] = 1;
	}
	for(size_t i = 0; i < ndims; i++)
		dims[i] = (new_coords[i] == d->dims[i] - 1 ?
			      d->border_tile_dims[i] :
			      d->tile_dims[i] );
	return d->l->ops->aslice_column(d->l->data, offsets, dims, strides);
}

struct aml_layout*
aml_tiling_nd_collapse_row_index(const struct aml_tiling_nd_data *l,
				  va_list coords)
{
	const struct aml_tiling_nd_data_collapse *d =
	    (const struct aml_tiling_nd_data_collapse *)l;
	size_t n_coords[d->ndims];
	for(size_t i = 0, j = 0; i < d->ndims; i++)
		if (d->dims[i] > 1)
			n_coords[j++] = va_arg(coords, size_t);
	return aml_tiling_nd_collapse_row_aindex(l, n_coords);
}

int
aml_tiling_nd_collapse_row_order(const struct aml_tiling_nd_data * l)
{
	return AML_TYPE_TILING_ROW_ORDER;
}

int
aml_tiling_nd_collapse_row_tile_dims(const struct aml_tiling_nd_data *l,
				      va_list dims_ptrs)
{
	const struct aml_tiling_nd_data_collapse *d =
	    (const struct aml_tiling_nd_data_collapse *)l;
	assert(d != NULL);
	for(size_t i = 0; i < d->ndims; i++) {
		size_t *dim = va_arg(dims_ptrs, size_t*);
		assert(dim != NULL);
		*dim = d->tile_dims[d->ndims - i - 1];
	}
	return 0;
}

int
aml_tiling_nd_collapse_row_tile_adims(const struct aml_tiling_nd_data *l,
				       size_t *tile_dims)
{
	const struct aml_tiling_nd_data_collapse *d =
	    (const struct aml_tiling_nd_data_collapse *)l;
	assert(d != NULL);
	for(size_t i = 0; i < d->ndims; i++) {
		tile_dims[i] = d->tile_dims[d->ndims - i - 1];
	}
	return 0;	
}

int
aml_tiling_nd_collapse_row_dims(const struct aml_tiling_nd_data *l,
				 va_list dims_ptrs)
{
	const struct aml_tiling_nd_data_collapse *d =
	    (const struct aml_tiling_nd_data_collapse *)l;
	assert(d != NULL);
	for(size_t i = 0; i < d->ndims; i++) {
		if (d->dims[i] > 1) {
			size_t *dim = va_arg(dims_ptrs, size_t*);
			assert(dim != NULL);
			*dim = d->dims[d->ndims - i - 1];
		}
	}
	return 0;
}

int
aml_tiling_nd_collapse_row_adims(const struct aml_tiling_nd_data *l,
				  size_t *dims)
{
	const struct aml_tiling_nd_data_collapse *d =
	    (const struct aml_tiling_nd_data_collapse *)l;
	assert(d != NULL);
	for(size_t i = 0, j = 0; i < d->ndims; i++)
		if (d->dims[i] > 1)
			dims[j++] = d->dims[d->ndims - i - 1];
	return 0;	
}

size_t
aml_tiling_nd_collapse_row_ndims(const struct aml_tiling_nd_data *l)
{
	const struct aml_tiling_nd_data_collapse *d =
	    (const struct aml_tiling_nd_data_collapse *)l;
	assert(d != NULL);
	size_t ndims = 0;
	for(size_t i = 0; i < d->ndims; i++)
		if (d->dims[i] > 1)
			ndims++;
	return ndims;
}

struct aml_tiling_nd_ops aml_tiling_nd_collapse_row_ops = {
	aml_tiling_nd_collapse_row_index,
	aml_tiling_nd_collapse_row_aindex,
	aml_tiling_nd_collapse_row_order,
	aml_tiling_nd_collapse_row_tile_dims,
	aml_tiling_nd_collapse_row_tile_adims,
	aml_tiling_nd_collapse_row_dims,
	aml_tiling_nd_collapse_row_adims,
	aml_tiling_nd_collapse_row_ndims
};
