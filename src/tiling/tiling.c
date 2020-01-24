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
#include "aml/tiling/native.h"
#include <assert.h>

int aml_tiling_order(const struct aml_tiling *t)
{
	if (t == NULL || t->ops == NULL)
		return -AML_EINVAL;

	return t->ops->order(t->data);
}

int aml_tiling_tile_dims(const struct aml_tiling *t, size_t *dims)
{
	if (t == NULL || t->ops == NULL || dims == NULL)
		return -AML_EINVAL;

	return t->ops->tile_dims(t->data, dims);
}

int aml_tiling_dims(const struct aml_tiling *t, size_t *dims)
{
	if (t == NULL || t->ops == NULL || dims == NULL)
		return -AML_EINVAL;

	return t->ops->dims(t->data, dims);
}

int aml_tiling_dims_native(const struct aml_tiling *t, size_t *dims)
{
	if (t == NULL || t->ops == NULL || dims == NULL)
		return -AML_EINVAL;

	return t->ops->dims_native(t->data, dims);
}

size_t aml_tiling_ndims(const struct aml_tiling *t)
{
	assert(t != NULL && t->ops != NULL);
	return t->ops->ndims(t->data);
}

size_t aml_tiling_ntiles(const struct aml_tiling *t)
{
	assert(t != NULL && t->ops != NULL);
	return t->ops->ntiles(t->data);
}

struct aml_layout *aml_tiling_index(const struct aml_tiling *t,
				    const size_t *coords)
{
	if (t == NULL || t->ops == NULL || coords == NULL)
		return NULL;

	return t->ops->index(t->data, coords);
}

void *aml_tiling_rawptr(const struct aml_tiling *t, const size_t *coords)
{
	if (t == NULL || t->ops == NULL || coords == NULL)
		return NULL;

	return t->ops->rawptr(t->data, coords);
}

int aml_tiling_tileid(const struct aml_tiling *t,
		      const size_t *coords)
{
	if (t == NULL || t->ops == NULL || coords == NULL)
		return -AML_EINVAL;

	return t->ops->tileid(t->data, coords);
}


struct aml_layout *aml_tiling_index_native(const struct aml_tiling *t,
					   const size_t *coords)
{
	if (t == NULL || t->ops == NULL || coords == NULL)
		return NULL;

	return t->ops->index_native(t->data, coords);
}

struct aml_layout *aml_tiling_index_byid(const struct aml_tiling *t,
					 int uuid)
{
	if (t == NULL || t->ops == NULL || uuid < 0)
		return NULL;

	size_t ndims = aml_tiling_ndims(t);
	size_t coords[ndims];
	size_t dims[ndims];

	aml_tiling_dims_native(t, dims);
	for (size_t i = 0; i < ndims; i++) {
		coords[i] = uuid % dims[i];
		uuid /= dims[i];
	}
	return aml_tiling_index_native(t, coords);
}

