/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
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

int aml_tiling_tile_dims(const struct aml_tiling *t,
                         const size_t *coords,
                         size_t *dims)
{
	if (t == NULL || t->ops == NULL || dims == NULL)
		return -AML_EINVAL;

	int err;
	struct aml_layout *layout;
	size_t ndims = aml_tiling_ndims(t);
	size_t coordinates[ndims];

	if (coords != NULL)
		memcpy(coordinates, coords, ndims * sizeof(*coords));
	else
		for (size_t i = 0; i < ndims; i++)
			coordinates[i] = 0;
	layout = aml_tiling_index(t, coordinates);
	if (layout == NULL)
		return -aml_errno;
	err = aml_layout_dims(layout, dims);
	aml_layout_destroy(&layout);

	return err;
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

struct aml_layout *aml_tiling_index_native(const struct aml_tiling *t,
					   const size_t *coords)
{
	if (t == NULL || t->ops == NULL || coords == NULL)
		return NULL;

	return t->ops->index_native(t->data, coords);
}

struct aml_layout *aml_tiling_index_byiter(const struct aml_tiling *t,
                                           const_excit_t iterator)
{
	if (t == NULL || t->ops == NULL || iterator == NULL)
		return NULL;

	ssize_t ncoords;
	size_t ndims = aml_tiling_ndims(t);

	assert(!excit_dimension(iterator, &ncoords));
	if ((size_t)ncoords != ndims)
		return NULL;

	ssize_t coords[ncoords];
	assert(!excit_peek(iterator, coords));

	return aml_tiling_index_native(t, (size_t *)coords);
}

int aml_tiling_fprintf(FILE *stream, const char *prefix,
		       const struct aml_tiling *tiling)
{
	assert(tiling != NULL && tiling->ops != NULL && stream != NULL);

	const char *p = (prefix == NULL) ? "" : prefix;

	return tiling->ops->fprintf(tiling->data, stream, p);
}
