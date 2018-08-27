#include <aml.h>
#include <assert.h>

/*******************************************************************************
 * 2D Iterator
 ******************************************************************************/

int aml_tiling_iterator_2d_reset(struct aml_tiling_iterator_data *data)
{
	struct aml_tiling_iterator_2d_data *it =
		(struct aml_tiling_iterator_2d_data *)data;
	it->i = 0;
	return 0;
}

int aml_tiling_iterator_2d_end(const struct aml_tiling_iterator_data *data)
{
	const struct aml_tiling_iterator_2d_data *it =
		(const struct aml_tiling_iterator_2d_data *)data;
	return it->i * it->tiling->blocksize >= it->tiling->totalsize;
}

int aml_tiling_iterator_2d_next(struct aml_tiling_iterator_data *data)
{
	struct aml_tiling_iterator_2d_data *it =
		(struct aml_tiling_iterator_2d_data *)data;
	it->i++;
	return 0;
}

int aml_tiling_iterator_2d_get(const struct aml_tiling_iterator_data *data,
			       va_list args)
{
	const struct aml_tiling_iterator_2d_data *it =
		(const struct aml_tiling_iterator_2d_data *)data;
	unsigned long *x = va_arg(args, unsigned long *);
	*x = it->i;
	return 0;
}

struct aml_tiling_iterator_ops aml_tiling_iterator_2d_ops = {
	aml_tiling_iterator_2d_reset,
	aml_tiling_iterator_2d_next,
	aml_tiling_iterator_2d_end,
	aml_tiling_iterator_2d_get,
};

/*******************************************************************************
 * 2D ops
 * Tileids are the order in memory of the tiles.
 ******************************************************************************/

int aml_tiling_2d_rowmajor_tileid(const struct aml_tiling_data *t, va_list ap)
{
	const struct aml_tiling_2d_data *data =
		(const struct aml_tiling_2d_data *)t;
	size_t row = va_arg(ap, size_t);
	size_t col = va_arg(ap, size_t);
	if(row >= data->ndims[0] || col >= data->ndims[1])
		return -1;
	else
		return (row*data->ndims[1]) + col;
}

int aml_tiling_2d_colmajor_tileid(const struct aml_tiling_data *t, va_list ap)
{
	const struct aml_tiling_2d_data *data =
		(const struct aml_tiling_2d_data *)t;
	size_t row = va_arg(ap, size_t);
	size_t col = va_arg(ap, size_t);
	if(row >= data->ndims[0] || col >= data->ndims[1])
		return -1;
	else
		return (col*data->ndims[0]) + row;
}

size_t aml_tiling_2d_tilesize(const struct aml_tiling_data *t, int tileid)
{
	const struct aml_tiling_2d_data *data =
		(const struct aml_tiling_2d_data *)t;
	if(tileid < 0 || tileid >= data->ndims[0]*data->ndims[1])
		return 0;
	else
		return data->blocksize;
}

void* aml_tiling_2d_tilestart(const struct aml_tiling_data *t,
			      const void *ptr, int tileid)
{
	const struct aml_tiling_2d_data *data =
		(const struct aml_tiling_2d_data *)t;
	intptr_t p = (intptr_t)ptr;
	if(tileid < 0 || tileid >= data->ndims[0]*data->ndims[1])
		return NULL;
	else
		return (void *)(p + tileid*data->blocksize);
}

int aml_tiling_2d_ndims(const struct aml_tiling_data *t, va_list ap)
{
	const struct aml_tiling_2d_data *data =
		(const struct aml_tiling_2d_data *)t;
	size_t *nrows = va_arg(ap, size_t *);
	size_t *ncols = va_arg(ap, size_t *);
	/* looks totally wrong */
	*nrows = data->ndims[0];
	*ncols = data->ndims[1];
	return 0;
}

int aml_tiling_2d_init_iterator(struct aml_tiling_data *t,
				struct aml_tiling_iterator *it, int flags)
{
	assert(it->data != NULL);
	struct aml_tiling_iterator_2d_data *data =
		(struct aml_tiling_iterator_2d_data *)it->data;
	it->ops = &aml_tiling_iterator_2d_ops;
	data->i = 0;
	data->tiling = (struct aml_tiling_2d_data *)t;
	return 0;
}

int aml_tiling_2d_create_iterator(struct aml_tiling_data *t,
				  struct aml_tiling_iterator **it, int flags)
{
	intptr_t baseptr, dataptr;
	struct aml_tiling_iterator *ret;
	baseptr = (intptr_t) calloc(1, AML_TILING_ITERATOR_2D_ALLOCSIZE);
	dataptr = baseptr + sizeof(struct aml_tiling_iterator);

	ret = (struct aml_tiling_iterator *)baseptr;
	ret->data = (struct aml_tiling_iterator_data *)dataptr;

	aml_tiling_2d_init_iterator(t, ret, flags);
	*it = ret;
	return 0;
}


int aml_tiling_2d_destroy_iterator(struct aml_tiling_data *t,
				   struct aml_tiling_iterator *it)
{
	return 0;
}


struct aml_tiling_ops aml_tiling_2d_rowmajor_ops = {
	aml_tiling_2d_create_iterator,
	aml_tiling_2d_init_iterator,
	aml_tiling_2d_destroy_iterator,
	aml_tiling_2d_rowmajor_tileid,
	aml_tiling_2d_tilesize,
	aml_tiling_2d_tilestart,
	aml_tiling_2d_ndims,
};

struct aml_tiling_ops aml_tiling_2d_colmajor_ops = {
	aml_tiling_2d_create_iterator,
	aml_tiling_2d_init_iterator,
	aml_tiling_2d_destroy_iterator,
	aml_tiling_2d_colmajor_tileid,
	aml_tiling_2d_tilesize,
	aml_tiling_2d_tilestart,
	aml_tiling_2d_ndims,
};
