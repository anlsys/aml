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
#include "aml/tiling/2d.h"
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

	if (row >= data->ndims[0] || col >= data->ndims[1])
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

	if (row >= data->ndims[0] || col >= data->ndims[1])
		return -1;
	else
		return (col*data->ndims[0]) + row;
}

size_t aml_tiling_2d_tilesize(const struct aml_tiling_data *t, int tileid)
{
	const struct aml_tiling_2d_data *data =
		(const struct aml_tiling_2d_data *)t;

	if (tileid < 0 || tileid >= (int)(data->ndims[0]*data->ndims[1]))
		return 0;
	else
		return data->blocksize;
}

void *aml_tiling_2d_tilestart(const struct aml_tiling_data *t,
			      const void *ptr, int tileid)
{
	const struct aml_tiling_2d_data *data =
		(const struct aml_tiling_2d_data *)t;
	intptr_t p = (intptr_t)ptr;

	if (tileid < 0 || tileid >= (int)(data->ndims[0]*data->ndims[1]))
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
	(void)flags;
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

int aml_tiling_2d_fini_iterator(struct aml_tiling_data *t,
				struct aml_tiling_iterator *it)
{
	(void)t;
	(void)it;
	return 0;
}

int aml_tiling_2d_destroy_iterator(struct aml_tiling_data *t,
				   struct aml_tiling_iterator **it)
{
	(void)t;
	free(*it);
	return 0;
}

struct aml_tiling_ops aml_tiling_2d_rowmajor_ops = {
	aml_tiling_2d_create_iterator,
	aml_tiling_2d_init_iterator,
	aml_tiling_2d_fini_iterator,
	aml_tiling_2d_destroy_iterator,
	aml_tiling_2d_rowmajor_tileid,
	aml_tiling_2d_tilesize,
	aml_tiling_2d_tilestart,
	aml_tiling_2d_ndims,
};

struct aml_tiling_ops aml_tiling_2d_colmajor_ops = {
	aml_tiling_2d_create_iterator,
	aml_tiling_2d_init_iterator,
	aml_tiling_2d_fini_iterator,
	aml_tiling_2d_destroy_iterator,
	aml_tiling_2d_colmajor_tileid,
	aml_tiling_2d_tilesize,
	aml_tiling_2d_tilestart,
	aml_tiling_2d_ndims,
};

/*******************************************************************************
 * 2d create/destroy
 ******************************************************************************/

int aml_tiling_2d_create(struct aml_tiling **t, int type,
			 size_t tilesize, size_t totalsize,
			 size_t rowsize, size_t colsize)
{
	struct aml_tiling *ret = NULL;
	intptr_t baseptr, dataptr;
	int err;

	if (t == NULL)
		return -AML_EINVAL;

	if (type != AML_TILING_TYPE_2D_ROWMAJOR &&
	    type != AML_TILING_TYPE_2D_COLMAJOR)
		return -AML_EINVAL;

	/* alloc */
	baseptr = (intptr_t) calloc(1, AML_TILING_2D_ALLOCSIZE);
	if (baseptr == 0) {
		*t = NULL;
		return -AML_ENOMEM;
	}
	dataptr = baseptr + sizeof(struct aml_tiling);

	ret = (struct aml_tiling *)baseptr;
	ret->data = (struct aml_tiling_data *)dataptr;
	if (type == AML_TILING_TYPE_2D_ROWMAJOR)
		ret->ops = &aml_tiling_2d_rowmajor_ops;
	else
		ret->ops = &aml_tiling_2d_colmajor_ops;

	err = aml_tiling_2d_init(ret, type, tilesize, totalsize,
				 rowsize, colsize);
	if (err) {
		free(ret);
		ret = NULL;
	}

	*t = ret;
	return err;
}


int aml_tiling_2d_init(struct aml_tiling *t, int type,
		       size_t tilesize, size_t totalsize,
		       size_t rowsize, size_t colsize)
{
	struct aml_tiling_2d_data *data;
	(void)type;

	if (t == NULL || t->data == NULL)
		return -AML_EINVAL;
	data = (struct aml_tiling_2d_data *)t->data;

	if (tilesize > totalsize)
		return -AML_EINVAL;

	data->blocksize = tilesize;
	data->totalsize = totalsize;
	data->ndims[0] = rowsize;
	data->ndims[1] = colsize;
	return 0;
}

void aml_tiling_2d_fini(struct aml_tiling *t)
{
	/* nothing to do */
	(void)t;
}


void aml_tiling_2d_destroy(struct aml_tiling **t)
{
	if (t == NULL)
		return;
	free(*t);
	*t = NULL;
}

