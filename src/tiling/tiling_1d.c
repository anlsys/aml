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
	(void)t;
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
		*x += 1;
	return 0;
}

int aml_tiling_1d_init_iterator(struct aml_tiling_data *t,
				struct aml_tiling_iterator *it, int flags)
{
	assert(it->data != NULL);
	(void)flags;
	struct aml_tiling_iterator_1d_data *data =
		(struct aml_tiling_iterator_1d_data *)it->data;

	it->ops = &aml_tiling_iterator_1d_ops;
	data->i = 0;
	data->tiling = (struct aml_tiling_1d_data *)t;
	return 0;
}

int aml_tiling_1d_create_iterator(struct aml_tiling_data *tiling,
				  struct aml_tiling_iterator **it, int flags)
{
	struct aml_tiling_iterator *ret;
	struct aml_tiling_iterator_1d_data *data;
	(void)flags;

	if (it == NULL)
		return -AML_EINVAL;

	*it = NULL;

	ret = AML_INNER_MALLOC_2(struct aml_tiling_iterator,
				 struct aml_tiling_iterator_1d_data);
	if (ret == NULL)
		return -AML_ENOMEM;

	ret->ops = &aml_tiling_iterator_1d_ops;
	ret->data = AML_INNER_MALLOC_NEXTPTR(ret, struct aml_tiling_iterator,
					struct aml_tiling_iterator_1d_data);
	data = (struct aml_tiling_iterator_1d_data *)ret->data;
	data->i = 0;
	data->tiling = (struct aml_tiling_1d_data *)tiling;
	*it = ret;
	return AML_SUCCESS;
}

int aml_tiling_1d_destroy_iterator(struct aml_tiling_data *t,
				   struct aml_tiling_iterator **iter)
{
	struct aml_tiling_iterator *it;
	(void)t;

	if (iter == NULL)
		return -AML_EINVAL;
	it = *iter;
	if (it == NULL)
		return -AML_EINVAL;
	free(it);
	*iter = NULL;
	return AML_SUCCESS;
}

struct aml_tiling_ops aml_tiling_1d_ops = {
	aml_tiling_1d_create_iterator,
	aml_tiling_1d_destroy_iterator,
	aml_tiling_1d_tileid,
	aml_tiling_1d_tilesize,
	aml_tiling_1d_tilestart,
	aml_tiling_1d_ndims,
};

/*******************************************************************************
 * 1D create/destroy
 ******************************************************************************/

int aml_tiling_1d_create(struct aml_tiling **tiling,
			 size_t tilesize, size_t totalsize)
{
	struct aml_tiling *ret = NULL;
	struct aml_tiling_1d_data *t;

	if (tiling == NULL || tilesize > totalsize)
		return -AML_EINVAL;

	*tiling = NULL;

	ret = AML_INNER_MALLOC_2(struct aml_tiling, struct aml_tiling_1d_data);
	if (ret == NULL)
		return -AML_ENOMEM;

	ret->ops = &aml_tiling_1d_ops;
	ret->data = AML_INNER_MALLOC_NEXTPTR(ret, struct aml_tiling,
					     struct aml_tiling_1d_data);
	t = (struct aml_tiling_1d_data *) ret->data;

	t->blocksize = tilesize;
	t->totalsize = totalsize;

	*tiling = ret;
	return AML_SUCCESS;
}

void aml_tiling_1d_destroy(struct aml_tiling **tiling)
{
	struct aml_tiling *t;

	if (tiling == NULL)
		return;
	t = *tiling;
	if (t == NULL)
		return;
	free(t);
	*tiling = NULL;
}

