#include <aml.h>
#include <assert.h>

/*******************************************************************************
 * Tiling functions
 ******************************************************************************/
size_t aml_tiling_vtilesize(struct aml_tiling *t, va_list args)
{
	assert(t != NULL);
	return t->ops->tilesize(t->data, args);
}

size_t aml_tiling_tilesize(struct aml_tiling *t, ...)
{
	assert(t != NULL);
	va_list ap;
	size_t ret;
	va_start(ap, t);
	ret = aml_tiling_vtilesize(t, ap);
	va_end(ap);
	return ret;
}

void* aml_tiling_vtilestart(struct aml_tiling *t, void *ptr, va_list args)
{
	assert(t != NULL);
	return t->ops->tilestart(t->data, ptr, args);
}

void* aml_tiling_tilestart(struct aml_tiling *t, void *ptr, ...)
{
	assert(t != NULL);
	va_list ap;
	void* ret;
	va_start(ap, ptr);
	ret = aml_tiling_vtilestart(t, ptr, ap);
	va_end(ap);
	return ret;
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

int aml_tiling_iterator_end(struct aml_tiling_iterator *it)
{
	assert(it != NULL);
	return it->ops->end(it->data);
}

int aml_tiling_iterator_get(struct aml_tiling_iterator *it, ...)
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
int aml_tiling_create(struct aml_tiling **t, int type, ...)
{
	va_list ap;
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
	
		aml_tiling_vinit(ret, type, ap);
	}
	va_end(ap);
	*t = ret;
	return 0;
}

int aml_tiling_vinit(struct aml_tiling *t, int type, va_list ap)
{
	if(type == AML_TILING_TYPE_1D)
	{
		t->ops = &aml_tiling_1d_ops;
		struct aml_tiling_1d_data *data = 
			(struct aml_tiling_1d_data *)t->data;
		data->blocksize = va_arg(ap, size_t);
		data->totalsize = va_arg(ap, size_t);
	}
	return 0;
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

