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

#include "aml/layout/sparse.h"

int aml_layout_sparse_create(struct aml_layout **layout,
                             const size_t nptr,
                             void **ptrs,
                             const size_t *sizes,
                             void *metadata,
                             const size_t metadata_size)
{
	struct aml_layout *ret;
	struct aml_layout_sparse *data;

	ret = AML_INNER_MALLOC_EXTRA(nptr, void *,
	                             nptr * sizeof(size_t) + metadata_size,
	                             struct aml_layout,
	                             struct aml_layout_sparse);
	if (ret == NULL)
		return -AML_ENOMEM;

	data = AML_INNER_MALLOC_GET_FIELD(ret, 2, struct aml_layout,
	                                  struct aml_layout_sparse);
	data->ptrs = AML_INNER_MALLOC_GET_ARRAY(ret, void *, struct aml_layout,
	                                        struct aml_layout_sparse);
	data->sizes = AML_INNER_MALLOC_GET_EXTRA(
	        ret, nptr, void *, struct aml_layout, struct aml_layout_sparse);
	data->metadata_size = metadata_size;
	memcpy(data->ptrs, ptrs, nptr * sizeof(*ptrs));
	memcpy(data->sizes, sizes, nptr * sizeof(*sizes));
	if (metadata_size > 0) {
		data->metadata =
		        data->sizes + nptr * sizeof(size_t) + sizeof(size_t);
		memcpy(data->metadata, metadata, metadata_size);
	} else
		data->metadata = NULL;
	data->nptr = nptr;

	ret->data = (struct aml_layout_data *)data;
	ret->ops = &aml_layout_sparse_ops;
	*layout = ret;
	return AML_SUCCESS;
}

int aml_layout_sparse_duplicate(const struct aml_layout *layout,
                                struct aml_layout **dest)
{
	struct aml_layout_sparse *src =
	        (struct aml_layout_sparse *)layout->data;
	return aml_layout_sparse_create(dest, src->nptr, src->ptrs, src->sizes,
	                                src->metadata, src->metadata_size);
}

void *aml_layout_sparse_deref(const struct aml_layout_data *data,
                              const size_t *coords)
{
	struct aml_layout_sparse *layout = (struct aml_layout_sparse *)data;
	return layout->ptrs[*coords];
}

int aml_layout_sparse_order(const struct aml_layout_data *data)
{
	(void)data;
	return AML_LAYOUT_ORDER_ROW_MAJOR;
}

size_t aml_layout_sparse_ndims(const struct aml_layout_data *data)
{
	(void)data;
	return 1;
}

int aml_layout_sparse_dims(const struct aml_layout_data *data, size_t *dims)
{
	struct aml_layout_sparse *layout = (struct aml_layout_sparse *)data;
	*dims = layout->nptr;
	return AML_SUCCESS;
}

size_t aml_layout_sparse_element_size(const struct aml_layout_data *data)
{
	(void)data;
	return sizeof(void *);
}

int aml_layout_sparse_fprintf(const struct aml_layout_data *data,
                              FILE *stream,
                              const char *prefix)
{
	struct aml_layout_sparse *layout = (struct aml_layout_sparse *)data;
	fprintf(stream, "%s[", prefix);
	for (size_t i = 0; i < layout->nptr - 1; i++)
		fprintf(stream, "%p, ", layout->ptrs[i]);
	fprintf(stream, "%p]\n", layout->ptrs[layout->nptr - 1]);
	return AML_SUCCESS;
}

struct aml_layout_ops aml_layout_sparse_ops = {
        .deref = aml_layout_sparse_deref,
        .deref_native = aml_layout_sparse_deref,
        .rawptr = NULL,
        .order = aml_layout_sparse_order,
        .dims = aml_layout_sparse_dims,
        .dims_native = aml_layout_sparse_dims,
        .ndims = aml_layout_sparse_ndims,
        .element_size = aml_layout_sparse_element_size,
        .reshape = NULL,
        .slice = NULL,
        .slice_native = NULL,
        .fprintf = aml_layout_sparse_fprintf,
        .duplicate = aml_layout_sparse_duplicate,
        .destroy = NULL,
};
