/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
*******************************************************************************/

#include "aml.h"
#include <assert.h>

/*******************************************************************************
 * Tiling functions
 ******************************************************************************/
int aml_tiling_tileid(const struct aml_tiling *t, ...)
{
	assert(t != NULL);
	va_list ap;
	int ret;
	va_start(ap, t);
	ret = t->ops->tileid(t->data, ap);
	va_end(ap);
	return ret;
}

size_t aml_tiling_tilesize(const struct aml_tiling *t, int tileid)
{
	assert(t != NULL);
	return t->ops->tilesize(t->data, tileid);
}

void* aml_tiling_tilestart(const struct aml_tiling *t, const void *ptr, int tileid)
{
	assert(t != NULL);
	return t->ops->tilestart(t->data, ptr, tileid);
}

int aml_tiling_ndims(const struct aml_tiling *t, ...)
{
	assert(t != NULL);
	va_list ap;
	int err;
	va_start(ap, t);
	err = t->ops->ndims(t->data, ap);
	va_end(ap);
	return err;
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
	else if(type == AML_TILING_TYPE_2D_ROWMAJOR ||
		type == AML_TILING_TYPE_2D_COLMAJOR)
	{
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
	else if(type == AML_TILING_TYPE_2D_ROWMAJOR)
	{
		t->ops = &aml_tiling_2d_rowmajor_ops;
		struct aml_tiling_2d_data *data =
			(struct aml_tiling_2d_data *)t->data;
		data->blocksize = va_arg(ap, size_t);
		data->totalsize = va_arg(ap, size_t);
		data->ndims[0] = va_arg(ap, size_t);
		data->ndims[1] = va_arg(ap, size_t);
		err = data->blocksize > data->totalsize;
	}
	else if(type == AML_TILING_TYPE_2D_COLMAJOR)
	{
		t->ops = &aml_tiling_2d_colmajor_ops;
		struct aml_tiling_2d_data *data =
			(struct aml_tiling_2d_data *)t->data;
		data->blocksize = va_arg(ap, size_t);
		data->totalsize = va_arg(ap, size_t);
		data->ndims[0] = va_arg(ap, size_t);
		data->ndims[1] = va_arg(ap, size_t);
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

