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
#include "aml/layout/dense.h"
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

static inline void aml_copy_shndstr_helper(size_t d, const size_t * target_dims,
					   void *dst,
					   const size_t * cumul_dst_pitch,
					   const size_t * dst_stride,
					   const void *src,
					   const size_t * cumul_src_pitch,
					   const size_t * src_stride,
					   const size_t * elem_number,
					   size_t elem_size)
{
	if (d == 1)
		if (dst_stride[0] * cumul_dst_pitch[0] == elem_size
		    && src_stride[target_dims[0]] *
		    cumul_src_pitch[target_dims[0]] == elem_size)
			memcpy(dst, src,
			       elem_number[target_dims[0]] * elem_size);
		else
			for (size_t i = 0; i < elem_number[target_dims[0]];
			     i += 1)
				memcpy((void *)((intptr_t) dst +
						i * (dst_stride[0] *
						     cumul_dst_pitch[0])),
				       (void *)((intptr_t) src +
						i *
						(src_stride[target_dims[0]] *
						 cumul_src_pitch[target_dims
								 [0]])),
				       elem_size);
	else
		for (size_t i = 0; i < elem_number[target_dims[d - 1]]; i += 1) {
			aml_copy_shndstr_helper(d - 1, target_dims, dst,
						cumul_dst_pitch, dst_stride,
						src, cumul_src_pitch,
						src_stride, elem_number,
						elem_size);
			dst =
			    (void *)((intptr_t) dst +
				     dst_stride[d - 1] * cumul_dst_pitch[d -
									 1]);
			src =
			    (void *)((intptr_t) src +
				     src_stride[target_dims[d - 1]] *
				     cumul_src_pitch[target_dims[d - 1]]);
		}
}

int aml_copy_shndstr_c(size_t d, const size_t * target_dims, void *dst,
		       const size_t * cumul_dst_pitch,
		       const size_t * dst_stride, const void *src,
		       const size_t * cumul_src_pitch,
		       const size_t * src_stride, const size_t * elem_number,
		       size_t elem_size)
{
	assert(d > 0);
	size_t present_dims;
	present_dims = 0;
	for (size_t i = 0; i < d; i += 1) {
		assert(target_dims[i] < d);
		present_dims |= 1 << target_dims[i];
	}
	for (size_t i = 0; i < d; i += 1)
		assert(present_dims & 1 << i);
	for (size_t i = 0; i < d - 1; i += 1) {
		assert(cumul_dst_pitch[i + 1] >=
		       dst_stride[i] * cumul_dst_pitch[i] *
		       elem_number[target_dims[i]]);
		assert(cumul_src_pitch[i + 1] >=
		       src_stride[i] * cumul_src_pitch[i] * elem_number[i]);
	}
	aml_copy_shndstr_helper(d, target_dims, dst, cumul_dst_pitch,
				dst_stride, src, cumul_src_pitch, src_stride,
				elem_number, elem_size);
	return 0;
}

int aml_copy_layout_transform_native(struct aml_layout *dst,
				     const struct aml_layout *src,
				     void *arg)
{
	size_t d;
	size_t elem_size;
	struct aml_layout_dense *ddst;
	struct aml_layout_dense *dsrc;
	const size_t *target_dims = (const size_t *)arg;
	ddst = (struct aml_layout_dense *)dst->data;
	dsrc = (struct aml_layout_dense *)src->data;
	d = dsrc->ndims;
	assert(d > 0);
	elem_size = dsrc->cpitch[0];
	assert(d == ddst->ndims);
	assert(elem_size == ddst->cpitch[0]);
	for (size_t i = 0; i < d; i += 1)
		assert(dsrc->dims[target_dims[i]] == ddst->dims[i]);
	return aml_copy_shndstr_c(d, target_dims, ddst->ptr, ddst->cpitch,
				  ddst->stride, dsrc->ptr, dsrc->cpitch,
				  dsrc->stride, dsrc->dims, elem_size);
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
