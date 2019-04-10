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
#include "aml/tiling/2d.h"
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

void *aml_tiling_tilestart(const struct aml_tiling *t, const void *ptr,
			   int tileid)
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
