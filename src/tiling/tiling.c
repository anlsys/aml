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

struct aml_layout *aml_tiling_index_linear(const struct aml_tiling *t,
					   size_t uuid)
{
	if (t == NULL || t->ops == NULL)
		return NULL;

	return t->ops->index_linear(t->data, uuid);
}

