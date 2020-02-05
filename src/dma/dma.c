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
#include "aml/layout/native.h"

#include <assert.h>

/*******************************************************************************
 * Generic DMA Copy implementations
 *
 * Needed by most DMAs. We don't provide introspection or any fancy API to it at
 * this point.
 ******************************************************************************/
static inline void aml_copy_layout_generic_helper(size_t d,
						  struct aml_layout *dst,
						  const struct aml_layout *src,
						  const size_t *elem_number,
						  size_t elem_size,
						  size_t *coords)
{
	if (d == 1) {
		for (size_t i = 0; i < elem_number[0]; i += 1) {
			coords[0] = i;
			memcpy(aml_layout_deref_native(dst, coords),
			       aml_layout_deref_native(src, coords),
			       elem_size);
		}
	} else {
		for (size_t i = 0; i < elem_number[d - 1]; i += 1) {
			coords[d - 1] = i;
			aml_copy_layout_generic_helper(d - 1, dst, src,
						       elem_number, elem_size,
						       coords);
		}
	}
}

int aml_copy_layout_generic(struct aml_layout *dst,
			    const struct aml_layout *src, void *arg)
{
	size_t d;
	size_t elem_size;
	(void)arg;

	assert(aml_layout_ndims(dst) == aml_layout_ndims(src));
	d = aml_layout_ndims(dst);
	assert(aml_layout_element_size(dst) == aml_layout_element_size(src));
	elem_size = aml_layout_element_size(dst);

	size_t coords[d];
	size_t elem_number[d];
	size_t elem_number2[d];

	aml_layout_dims_native(src, elem_number);
	aml_layout_dims_native(dst, elem_number2);
	for (size_t i = 0; i < d; i += 1)
		assert(elem_number[i] == elem_number2[i]);
	aml_copy_layout_generic_helper(d, dst, src, elem_number, elem_size,
				       coords);
	return 0;
}

static inline void aml_copy_layout_transform_generic_helper(
						size_t d,
						struct aml_layout *dst,
						const struct aml_layout *src,
						const size_t *elem_number,
						size_t elem_size,
						size_t *coords,
						size_t *coords_out,
						const size_t *target_dims)
{
	if (d == 1)
		for (size_t i = 0; i < elem_number[target_dims[0]]; i += 1) {
			coords_out[0] = i;
			coords[target_dims[0]] = i;
			memcpy(aml_layout_deref_native(dst, coords_out),
			       aml_layout_deref_native(src, coords),
			       elem_size);
	} else
		for (size_t i = 0; i < elem_number[target_dims[d-1]]; i += 1) {
			coords_out[d - 1] = i;
			coords[target_dims[d - 1]] = i;
			aml_copy_layout_transform_generic_helper(d - 1, dst,
								 src,
								 elem_number,
								 elem_size,
								 coords,
								 coords_out,
								 target_dims);
		}
}

int aml_copy_layout_transform_generic(struct aml_layout *dst,
				      const struct aml_layout *src,
				      const size_t *target_dims)
{
	size_t d;
	size_t elem_size;

	assert(aml_layout_ndims(dst) == aml_layout_ndims(src));
	d = aml_layout_ndims(dst);
	assert(aml_layout_element_size(dst) == aml_layout_element_size(src));
	elem_size = aml_layout_element_size(dst);

	size_t coords[d];
	size_t coords_out[d];
	size_t elem_number[d];
	size_t elem_number2[d];

	aml_layout_dims_native(src, elem_number);
	aml_layout_dims_native(dst, elem_number2);
	for (size_t i = 0; i < d; i += 1)
		assert(elem_number[target_dims[i]] == elem_number2[i]);
	aml_copy_layout_transform_generic_helper(d, dst, src, elem_number,
						 elem_size, coords, coords_out,
						 target_dims);
	return 0;
}

/*******************************************************************************
 * Generic DMA API:
 * Most of the stuff is dispatched to a different layer, using type-specific
 * functions.
 *
 * Note that the API is slightly different than the functions bellow, as we
 * abstract the request creation after this layer.
 ******************************************************************************/

int aml_dma_copy_custom(struct aml_dma *dma, struct aml_layout *dest,
		 struct aml_layout *src, aml_dma_operator op, void *op_arg)
{
	int ret;
	struct aml_dma_request *req;

	if (dma == NULL || dest == NULL || src == NULL)
		return -AML_EINVAL;

	ret = dma->ops->create_request(dma->data, &req, dest, src, op, op_arg);
	if (ret != AML_SUCCESS)
		return ret;
	ret = dma->ops->wait_request(dma->data, &req);
	return ret;
}

int aml_dma_async_copy_custom(struct aml_dma *dma, struct aml_dma_request **req,
		       struct aml_layout *dest, struct aml_layout *src,
		       aml_dma_operator op, void *op_arg)
{
	if (dma == NULL || req == NULL || dest == NULL || src == NULL)
		return -AML_EINVAL;

	return dma->ops->create_request(dma->data, req, dest, src, op, op_arg);
}

int aml_dma_cancel(struct aml_dma *dma, struct aml_dma_request **req)
{
	if (dma == NULL || req == NULL)
		return -AML_EINVAL;
	return dma->ops->destroy_request(dma->data, req);
}

int aml_dma_wait(struct aml_dma *dma, struct aml_dma_request **req)
{
	if (dma == NULL || req == NULL)
		return -AML_EINVAL;
	return dma->ops->wait_request(dma->data, req);
}

int aml_dma_fprintf(FILE *stream, const char *prefix,
		    const struct aml_dma *dma)
{
	assert(dma != NULL && dma->ops != NULL && stream != NULL);

	const char *p = (prefix == NULL) ? "" : prefix;

	return dma->ops->fprintf(dma->data, stream, p);
}
