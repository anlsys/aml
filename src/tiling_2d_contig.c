#include <aml.h>
#include <assert.h>

/*******************************************************************************
 * 2D Iterator
 ******************************************************************************/

int aml_tiling_iterator_2d_contig_reset(struct aml_tiling_iterator_data *data)
{
	struct aml_tiling_iterator_2d_contig_data *it =
		(struct aml_tiling_iterator_2d_contig_data *)data;
	it->i = 0;
	return 0;
}

int aml_tiling_iterator_2d_contig_end(const struct aml_tiling_iterator_data *data)
{
	const struct aml_tiling_iterator_2d_contig_data *it =
		(const struct aml_tiling_iterator_2d_contig_data *)data;
	return it->i * it->tiling->blocksize >= it->tiling->totalsize;
}

int aml_tiling_iterator_2d_contig_next(struct aml_tiling_iterator_data *data)
{
	struct aml_tiling_iterator_2d_contig_data *it =
		(struct aml_tiling_iterator_2d_contig_data *)data;
	it->i++;
	return 0;
}

int aml_tiling_iterator_2d_contig_get(const struct aml_tiling_iterator_data *data,
			       va_list args)
{
	const struct aml_tiling_iterator_2d_contig_data *it =
		(const struct aml_tiling_iterator_2d_contig_data *)data;
	unsigned long *x = va_arg(args, unsigned long *);
	*x = it->i;
	return 0;
}

struct aml_tiling_iterator_ops aml_tiling_iterator_2d_contig_ops = {
	aml_tiling_iterator_2d_contig_reset,
	aml_tiling_iterator_2d_contig_next,
	aml_tiling_iterator_2d_contig_end,
	aml_tiling_iterator_2d_contig_get,
};

/*******************************************************************************
 * 2D ops
 ******************************************************************************/

size_t aml_tiling_2d_contig_tilesize(const struct aml_tiling_data *t, int tileid)
{
	const struct aml_tiling_2d_contig_data *data =
		(const struct aml_tiling_2d_contig_data *)t;
	return data->blocksize;
}

void* aml_tiling_2d_contig_tilestart(const struct aml_tiling_data *t, const void *ptr, int tileid)
{
	const struct aml_tiling_2d_contig_data *data =
		(const struct aml_tiling_2d_contig_data *)t;
	intptr_t p = (intptr_t)ptr;
	return (void *)(p + tileid*data->blocksize);
}

int aml_tiling_2d_contig_ndims(const struct aml_tiling_data *t, va_list ap)
{
	const struct aml_tiling_2d_contig_data *data =
		(const struct aml_tiling_2d_contig_data *)t;
	size_t *x = va_arg(ap, size_t *);
	size_t *y = va_arg(ap, size_t *);
	/* looks totally wrong */
	*x = data->ndims[0];
	*y = data->ndims[1];
	return 0;
}

int aml_tiling_2d_contig_init_iterator(struct aml_tiling_data *t,
				struct aml_tiling_iterator *it, int flags)
{
	assert(it->data != NULL);
	struct aml_tiling_iterator_2d_contig_data *data =
		(struct aml_tiling_iterator_2d_contig_data *)it->data;
	it->ops = &aml_tiling_iterator_2d_contig_ops;
	data->i = 0;
	data->tiling = (struct aml_tiling_2d_contig_data *)t;
	return 0;
}

int aml_tiling_2d_contig_create_iterator(struct aml_tiling_data *t,
				  struct aml_tiling_iterator **it, int flags)
{
	intptr_t baseptr, dataptr;
	struct aml_tiling_iterator *ret;
	baseptr = (intptr_t) calloc(1, AML_TILING_ITERATOR_2D_CONTIG_ALLOCSIZE);
	dataptr = baseptr + sizeof(struct aml_tiling_iterator);

	ret = (struct aml_tiling_iterator *)baseptr;
	ret->data = (struct aml_tiling_iterator_data *)dataptr;

	aml_tiling_2d_contig_init_iterator(t, ret, flags);
	*it = ret;
	return 0;
}


int aml_tiling_2d_contig_destroy_iterator(struct aml_tiling_data *t,
				   struct aml_tiling_iterator *it)
{
	return 0;
}


struct aml_tiling_ops aml_tiling_2d_contig_ops = {
	aml_tiling_2d_contig_create_iterator,
	aml_tiling_2d_contig_init_iterator,
	aml_tiling_2d_contig_destroy_iterator,
	aml_tiling_2d_contig_tilesize,
	NULL,
	NULL,
	aml_tiling_2d_contig_tilestart,
	aml_tiling_2d_contig_ndims,
};
