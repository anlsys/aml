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
#include "aml/tiling/1d.h"
#include <assert.h>

/*******************************************************************************
 * 1D Iterator
 ******************************************************************************/

int aml_tiling_iterator_1d_reset(struct aml_tiling_iterator_data *data)
{
	struct aml_tiling_iterator_1d_data *it =
		(struct aml_tiling_iterator_1d_data *)data;
	it->i = 0;
	return 0;
}

int aml_tiling_iterator_1d_end(const struct aml_tiling_iterator_data *data)
{
	const struct aml_tiling_iterator_1d_data *it =
		(const struct aml_tiling_iterator_1d_data *)data;
	return it->i * it->tiling->blocksize >= it->tiling->totalsize;
}

int aml_tiling_iterator_1d_next(struct aml_tiling_iterator_data *data)
{
	struct aml_tiling_iterator_1d_data *it =
		(struct aml_tiling_iterator_1d_data *)data;
	it->i++;
	return 0;
}

int aml_tiling_iterator_1d_get(const struct aml_tiling_iterator_data *data,
			       va_list args)
{
	const struct aml_tiling_iterator_1d_data *it =
		(const struct aml_tiling_iterator_1d_data *)data;
	unsigned long *x = va_arg(args, unsigned long *);
	*x = it->i;
	return 0;
}

struct aml_tiling_iterator_ops aml_tiling_iterator_1d_ops = {
	aml_tiling_iterator_1d_reset,
	aml_tiling_iterator_1d_next,
	aml_tiling_iterator_1d_end,
	aml_tiling_iterator_1d_get,
};

/*******************************************************************************
 * 1D ops
 ******************************************************************************/

int aml_tiling_1d_tileid(const struct aml_tiling_data *t, va_list ap)
{
	const struct aml_tiling_1d_data *data =
		(const struct aml_tiling_1d_data *)t;
	size_t x = va_arg(ap, size_t);
	return x;
}

size_t aml_tiling_1d_tilesize(const struct aml_tiling_data *t, int tileid)
{
	const struct aml_tiling_1d_data *data =
		(const struct aml_tiling_1d_data *)t;

	if (tileid < 0)
		return 0;
	else
		return data->blocksize;
}

void *aml_tiling_1d_tilestart(const struct aml_tiling_data *t,
			      const void *ptr, int tileid)
{
	const struct aml_tiling_1d_data *data =
		(const struct aml_tiling_1d_data *)t;
	intptr_t p = (intptr_t)ptr;

	if (tileid < 0)
		return NULL;
	else
		return (void *)(p + tileid*data->blocksize);
}

int aml_tiling_1d_ndims(const struct aml_tiling_data *t, va_list ap)
{
	const struct aml_tiling_1d_data *data =
		(const struct aml_tiling_1d_data *)t;
	size_t *x = va_arg(ap, size_t *);
	*x = data->totalsize/data->blocksize;
	if (data->totalsize % data->blocksize != 0)
		*x++;
	return 0;
}

int aml_tiling_1d_init_iterator(struct aml_tiling_data *t,
				struct aml_tiling_iterator *it, int flags)
{
	assert(it->data != NULL);
	struct aml_tiling_iterator_1d_data *data =
		(struct aml_tiling_iterator_1d_data *)it->data;

	it->ops = &aml_tiling_iterator_1d_ops;
	data->i = 0;
	data->tiling = (struct aml_tiling_1d_data *)t;
	return 0;
}

int aml_tiling_1d_create_iterator(struct aml_tiling_data *t,
				  struct aml_tiling_iterator **it, int flags)
{
	intptr_t baseptr, dataptr;
	struct aml_tiling_iterator *ret;

	baseptr = (intptr_t) calloc(1, AML_TILING_ITERATOR_1D_ALLOCSIZE);
	dataptr = baseptr + sizeof(struct aml_tiling_iterator);

	ret = (struct aml_tiling_iterator *)baseptr;
	ret->data = (struct aml_tiling_iterator_data *)dataptr;

	aml_tiling_1d_init_iterator(t, ret, flags);
	*it = ret;
	return 0;
}

int aml_tiling_1d_destroy_iterator(struct aml_tiling_data *t,
				   struct aml_tiling_iterator *it)
{
	return 0;
}

struct aml_tiling_ops aml_tiling_1d_ops = {
	aml_tiling_1d_create_iterator,
	aml_tiling_1d_init_iterator,
	aml_tiling_1d_destroy_iterator,
	aml_tiling_1d_tileid,
	aml_tiling_1d_tilesize,
	aml_tiling_1d_tilestart,
	aml_tiling_1d_ndims,
};

/*******************************************************************************
 * 1D create/destroy
 ******************************************************************************/

int aml_tiling_1d_create(struct aml_tiling **t,
			 size_t tilesize, size_t totalsize)
{
	struct aml_tiling *ret = NULL;
	intptr_t baseptr, dataptr;
	int err;

	if (t == NULL)
		return -AML_EINVAL;

	/* alloc */
	baseptr = (intptr_t) calloc(1, AML_TILING_1D_ALLOCSIZE);
	if (baseptr == 0) {
		*t = NULL;
		return -AML_ENOMEM;
	}
	dataptr = baseptr + sizeof(struct aml_tiling);

	ret = (struct aml_tiling *)baseptr;
	ret->data = (struct aml_tiling_data *)dataptr;
	ret->ops = &aml_tiling_1d_ops;

	err = aml_tiling_1d_init(ret, tilesize, totalsize);
	if (err) {
		free(ret);
		ret = NULL;
	}

	*t = ret;
	return err;
}


int aml_tiling_1d_init(struct aml_tiling *t,
		       size_t tilesize, size_t totalsize)
{
	int err;
	struct aml_tiling_1d_data *data;

	if (t == NULL || t->data == NULL)
		return -AML_EINVAL;
	data = (struct aml_tiling_1d_data *)t->data;

	if (tilesize > totalsize)
		return -AML_EINVAL;

	data->blocksize = tilesize;
	data->totalsize = totalsize;
	return 0;
}

void aml_tiling_1d_fini(struct aml_tiling *t)
{
	/* nothing to do */
}


void aml_tiling_1d_destroy(struct aml_tiling **t)
{
	if (t == NULL)
		return;
	free(*t);
	*t = NULL;
}

