#include <aml.h>
#include <assert.h>

/*******************************************************************************
 * Tiling functions
 ******************************************************************************/
size_t aml_tiling_tilesize(const struct aml_tiling *t, int tileid)
{
	assert(t != NULL);
	return t->ops->tilesize(t->data, tileid);
}

size_t aml_tiling_tilerowsize(const struct aml_tiling *t, int tileid)
{
	assert(t != NULL);
	return t->ops->tilerowsize(t->data, tileid);
}

size_t aml_tiling_tilecolsize(const struct aml_tiling *t, int tileid)
{
	assert(t != NULL);
	return t->ops->tilecolsize(t->data, tileid);
}

void* aml_tiling_tilestart(const struct aml_tiling *t, const void *ptr, int tileid)
{
	assert(t != NULL);
	return t->ops->tilestart(t->data, ptr, tileid);
}

/*******************************************************************************
 * Tiling Iterator functions
 ******************************************************************************/

int aml_tiling_iterator_reset(struct aml_tiling_iterator *it)
{
	assert(it != NULL);
	return it->ops->reset(it->data);
}

int aml_tiling_iterator_next(struct aml_tiling_iterator *it)
{
	assert(it != NULL);
	return it->ops->next(it->data);
}

int aml_tiling_iterator_end(const struct aml_tiling_iterator *it)
{
	assert(it != NULL);
	return it->ops->end(it->data);
}

int aml_tiling_iterator_get(const struct aml_tiling_iterator *it, ...)
{
	assert(it != NULL);
	va_list ap;
	va_start(ap, it);
	it->ops->get(it->data, ap);
	va_end(ap);
	return 0;
}

/*******************************************************************************
 * Iterator Init
 * We can't do the allocation ourselves here, as we don't have the type of the
 * tiling.
 ******************************************************************************/

int aml_tiling_create_iterator(struct aml_tiling *t,
			       struct aml_tiling_iterator **it, int flags)
{
	assert(t != NULL);
	assert(it != NULL);
	return t->ops->create_iterator(t->data, it, flags);
}

int aml_tiling_init_iterator(struct aml_tiling *t,
			     struct aml_tiling_iterator *it, int flags)
{
	assert(t != NULL);
	assert(it != NULL);
	return t->ops->init_iterator(t->data, it, flags);
}

int aml_tiling_destroy_iterator(struct aml_tiling *t,
				struct aml_tiling_iterator *it)
{
	assert(t != NULL);
	assert(it != NULL);
	return t->ops->destroy_iterator(t->data, it);
}

/*******************************************************************************
 * Init functions
 ******************************************************************************/

/* allocate and init the tiling according to type */
//In the future, a n-dimensional arrya could be created with an arguments of:
//type: n # of dimensions
//va_list: size of each dimension followed by total size
//The return is now changed to ensure that a tile size is not larger than the given total size
int aml_tiling_create(struct aml_tiling **t, int type, ...)
{
	va_list ap;
	int err;
	va_start(ap, type);
	struct aml_tiling *ret = NULL;
	intptr_t baseptr, dataptr;
	if(type == AML_TILING_TYPE_1D)
	{
		/* alloc */
		baseptr = (intptr_t) calloc(1, AML_TILING_1D_ALLOCSIZE);
		dataptr = baseptr + sizeof(struct aml_tiling);

		ret = (struct aml_tiling *)baseptr;
		ret->data = (struct aml_tiling_data *)dataptr;

		err = aml_tiling_vinit(ret, type, ap);
	}
	else if(type == AML_TILING_TYPE_2D)
	{
		/* alloc, only difference is using AML_TILING_2D_ALLOCSIZE instead fo 1D */
		baseptr = (intptr_t) calloc(1, AML_TILING_2D_ALLOCSIZE);
		dataptr = baseptr + sizeof(struct aml_tiling);

		ret = (struct aml_tiling *)baseptr;
		ret->data = (struct aml_tiling_data *)dataptr;

		err = aml_tiling_vinit(ret, type, ap);
	}

	va_end(ap);
	*t = ret;
	return err;
}


int aml_tiling_vinit(struct aml_tiling *t, int type, va_list ap)
{
	int err;
	if(type == AML_TILING_TYPE_1D)
	{
		t->ops = &aml_tiling_1d_ops;
		struct aml_tiling_1d_data *data =
			(struct aml_tiling_1d_data *)t->data;
		data->blocksize = va_arg(ap, size_t);
		data->totalsize = va_arg(ap, size_t);
		err = data->blocksize > data->totalsize;
	}

	//This is equivalent to the 1D except the arguments will be the dimensions for the tile.
	//An optimization that could be made is having the exact same declaration of 1D
	//The caveat is that the block size must be a perfect square. For now, we will allow non-square blocks
	else if(type == AML_TILING_TYPE_2D)
	{
		t->ops = &aml_tiling_2d_ops;
		struct aml_tiling_2d_data *data =
			(struct aml_tiling_2d_data *)t->data;
		data->tilerowsize = va_arg(ap, size_t);
		data->tilecolsize = va_arg(ap, size_t);
		data->blocksize = data->tilerowsize * data->tilecolsize / sizeof(unsigned long);
		data->totalsize = va_arg(ap, size_t);
		err = data->blocksize > data->totalsize;
	}
	return err;
}

int aml_tiling_init(struct aml_tiling *t, int type, ...)
{
	int err;
	va_list ap;
	va_start(ap, type);
	err = aml_tiling_vinit(t, type, ap);
	va_end(ap);
	return err;
}

int aml_tiling_destroy(struct aml_tiling *b, int type)
{
	return 0;
}

