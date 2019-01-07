#include <aml.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <alloca.h>

static inline void aml_compute_cumulative_pitch(size_t d,
						size_t * cumul_dst_pitch,
						size_t * cumul_src_pitch,
						const size_t * dst_pitch,
						const size_t * src_pitch,
						size_t elem_size)
{
	cumul_dst_pitch[0] = elem_size;
	cumul_src_pitch[0] = elem_size;
	for (size_t i = 0; i < d - 1; i += 1) {
		cumul_dst_pitch[i + 1] = dst_pitch[i] * cumul_dst_pitch[i];
		cumul_src_pitch[i + 1] = src_pitch[i] * cumul_src_pitch[i];
	}
}

static inline void aml_copy_nd_helper(size_t d, void *dst,
				      const size_t * cumul_dst_pitch,
				      const void *src,
				      const size_t * cumul_src_pitch,
				      const size_t * elem_number,
				      size_t elem_size)
{
	if (d == 1)
		if (cumul_dst_pitch[0] == elem_size
		    && cumul_src_pitch[0] == elem_size)
			memcpy(dst, src, elem_number[0] * elem_size);
		else
			for (size_t i = 0; i < elem_number[0]; i += 1)
				memcpy((void *)((intptr_t) dst +
						i * cumul_dst_pitch[0]),
				       (void *)((intptr_t) src +
						i * cumul_src_pitch[0]),
				       elem_size);
	else
		for (size_t i = 0; i < elem_number[d - 1]; i += 1) {
			aml_copy_nd_helper(d - 1, dst, cumul_dst_pitch, src,
					   cumul_src_pitch, elem_number,
					   elem_size);
			dst = (void *)((intptr_t) dst + cumul_dst_pitch[d - 1]);
			src = (void *)((intptr_t) src + cumul_src_pitch[d - 1]);
		}
}

int aml_copy_nd_c(size_t d, void *dst, const size_t * cumul_dst_pitch,
		  const void *src, const size_t * cumul_src_pitch,
		  const size_t * elem_number, size_t elem_size)
{
	assert(d > 0);
	for (size_t i = 0; i < d - 1; i += 1) {
		assert(cumul_dst_pitch[i + 1] >=
		       cumul_dst_pitch[i] * elem_number[i]);
		assert(cumul_src_pitch[i + 1] >=
		       cumul_src_pitch[i] * elem_number[i]);
	}
	aml_copy_nd_helper(d, dst, cumul_dst_pitch, src, cumul_src_pitch,
			   elem_number, elem_size);
	return 0;
}

int aml_copy_nd(size_t d, void *dst, const size_t * dst_pitch, const void *src,
		const size_t * src_pitch, const size_t * elem_number,
		size_t elem_size)
{
	assert(d > 0);
	size_t *cumul_dst_pitch;
	size_t *cumul_src_pitch;
	cumul_dst_pitch = (size_t *) alloca(d * sizeof(size_t));
	cumul_src_pitch = (size_t *) alloca(d * sizeof(size_t));
	aml_compute_cumulative_pitch(d, cumul_dst_pitch, cumul_src_pitch,
				     dst_pitch, src_pitch, elem_size);
	aml_copy_nd_c(d, dst, cumul_dst_pitch, src, cumul_src_pitch,
		      elem_number, elem_size);
	return 0;
}

static inline void aml_copy_ndstr_helper(size_t d, void *dst,
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
		    && src_stride[0] * cumul_src_pitch[0] == elem_size)
			memcpy(dst, src, elem_number[0] * elem_size);
		else
			for (size_t i = 0; i < elem_number[0]; i += 1)
				memcpy((void *)((intptr_t) dst +
						i * (dst_stride[0] *
						     cumul_dst_pitch[0])),
				       (void *)((intptr_t) src +
						i * (src_stride[0] *
						     cumul_src_pitch[0])),
				       elem_size);
	else
		for (size_t i = 0; i < elem_number[d - 1]; i += 1) {
			aml_copy_ndstr_helper(d - 1, dst, cumul_dst_pitch,
					      dst_stride, src, cumul_src_pitch,
					      src_stride, elem_number,
					      elem_size);
			dst =
			    (void *)((intptr_t) dst +
				     dst_stride[d - 1] * cumul_dst_pitch[d -
									 1]);
			src =
			    (void *)((intptr_t) src +
				     src_stride[d - 1] * cumul_src_pitch[d -
									 1]);
		}
}

int aml_copy_ndstr_c(size_t d, void *dst, const size_t * cumul_dst_pitch,
		     const size_t * dst_stride, const void *src,
		     const size_t * cumul_src_pitch, const size_t * src_stride,
		     const size_t * elem_number, size_t elem_size)
{
	assert(d > 0);
	for (size_t i = 0; i < d - 1; i += 1) {
		assert(cumul_dst_pitch[i + 1] >=
		       dst_stride[i] * cumul_dst_pitch[i] * elem_number[i]);
		assert(cumul_src_pitch[i + 1] >=
		       src_stride[i] * cumul_src_pitch[i] * elem_number[i]);
	}
	aml_copy_ndstr_helper(d, dst, cumul_dst_pitch, dst_stride, src,
			      cumul_src_pitch, src_stride, elem_number,
			      elem_size);
	return 0;
}

int aml_copy_ndstr(size_t d, void *dst, const size_t * dst_pitch,
		   const size_t * dst_stride, const void *src,
		   const size_t * src_pitch, const size_t * src_stride,
		   const size_t * elem_number, size_t elem_size)
{
	assert(d > 0);
	size_t *cumul_dst_pitch;
	size_t *cumul_src_pitch;
	cumul_dst_pitch = (size_t *) alloca(d * sizeof(size_t));
	cumul_src_pitch = (size_t *) alloca(d * sizeof(size_t));
	aml_compute_cumulative_pitch(d, cumul_dst_pitch, cumul_src_pitch,
				     dst_pitch, src_pitch, elem_size);
	aml_copy_ndstr_c(d, dst, cumul_dst_pitch, dst_stride, src,
			 cumul_src_pitch, src_stride, elem_number, elem_size);
	return 0;
}

static inline void aml_copy_shnd_helper(size_t d, const size_t * target_dims,
					void *dst,
					const size_t * cumul_dst_pitch,
					const void *src,
					const size_t * cumul_src_pitch,
					const size_t * elem_number,
					size_t elem_size)
{
	if (d == 1)
		if (cumul_dst_pitch[0] == elem_size
		    && cumul_src_pitch[target_dims[0]] == elem_size)
			memcpy(dst, src,
			       elem_number[target_dims[0]] * elem_size);
		else
			for (size_t i = 0; i < elem_number[target_dims[0]];
			     i += 1)
				memcpy((void *)((intptr_t) dst +
						i * cumul_dst_pitch[0]),
				       (void *)((intptr_t) src +
						i *
						cumul_src_pitch[target_dims
								[0]]),
				       elem_size);
	else
		for (size_t i = 0; i < elem_number[target_dims[d - 1]]; i += 1) {
			aml_copy_shnd_helper(d - 1, target_dims, dst,
					     cumul_dst_pitch, src,
					     cumul_src_pitch, elem_number,
					     elem_size);
			dst = (void *)((intptr_t) dst + cumul_dst_pitch[d - 1]);
			src =
			    (void *)((intptr_t) src +
				     cumul_src_pitch[target_dims[d - 1]]);
		}
}

int aml_copy_shnd_c(size_t d, const size_t * target_dims, void *dst,
		    const size_t * cumul_dst_pitch, const void *src,
		    const size_t * cumul_src_pitch, const size_t * elem_number,
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
		       cumul_dst_pitch[i] * elem_number[target_dims[i]]);
		assert(cumul_src_pitch[i + 1] >=
		       cumul_src_pitch[i] * elem_number[i]);
	}
	aml_copy_shnd_helper(d, target_dims, dst, cumul_dst_pitch, src,
			     cumul_src_pitch, elem_number, elem_size);
	return 0;
}

int aml_copy_shnd(size_t d, const size_t * target_dims, void *dst,
		  const size_t * dst_pitch, const void *src,
		  const size_t * src_pitch, const size_t * elem_number,
		  size_t elem_size)
{
	assert(d > 0);
	size_t *cumul_dst_pitch;
	size_t *cumul_src_pitch;
	cumul_dst_pitch = (size_t *) alloca(d * sizeof(size_t));
	cumul_src_pitch = (size_t *) alloca(d * sizeof(size_t));
	aml_compute_cumulative_pitch(d, cumul_dst_pitch, cumul_src_pitch,
				     dst_pitch, src_pitch, elem_size);
	aml_copy_shnd_c(d, target_dims, dst, cumul_dst_pitch, src,
			cumul_src_pitch, elem_number, elem_size);
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

int aml_copy_shndstr(size_t d, const size_t * target_dims, void *dst,
		     const size_t * dst_pitch, const size_t * dst_stride,
		     const void *src, const size_t * src_pitch,
		     const size_t * src_stride, const size_t * elem_number,
		     size_t elem_size)
{
	assert(d > 0);
	size_t *cumul_dst_pitch;
	size_t *cumul_src_pitch;
	cumul_dst_pitch = (size_t *) alloca(d * sizeof(size_t));
	cumul_src_pitch = (size_t *) alloca(d * sizeof(size_t));
	aml_compute_cumulative_pitch(d, cumul_dst_pitch, cumul_src_pitch,
				     dst_pitch, src_pitch, elem_size);
	aml_copy_shndstr_c(d, target_dims, dst, cumul_dst_pitch, dst_stride,
			   src, cumul_src_pitch, src_stride, elem_number,
			   elem_size);
	return 0;
}

int aml_copy_tnd(size_t d, void *dst, const size_t * dst_pitch, const void *src,
		 const size_t * src_pitch, const size_t * elem_number,
		 size_t elem_size)
{
	assert(d > 0);
	size_t *target_dims;
	target_dims = (size_t *) alloca(d * sizeof(size_t));
	target_dims[d - 1] = 0;
	for (size_t i = 0; i < d - 1; i += 1)
		target_dims[i] = i + 1;
	aml_copy_shnd(d, target_dims, dst, dst_pitch, src, src_pitch,
		      elem_number, elem_size);
	return 0;
}

int aml_copy_tnd_c(size_t d, void *dst, const size_t * cumul_dst_pitch,
		   const void *src, const size_t * cumul_src_pitch,
		   const size_t * elem_number, size_t elem_size)
{
	assert(d > 0);
	size_t *target_dims;
	target_dims = (size_t *) alloca(d * sizeof(size_t));
	target_dims[d - 1] = 0;
	for (size_t i = 0; i < d - 1; i += 1)
		target_dims[i] = i + 1;
	aml_copy_shnd_c(d, target_dims, dst, cumul_dst_pitch, src,
			cumul_src_pitch, elem_number, elem_size);
	return 0;
}

int aml_copy_rtnd(size_t d, void *dst, const size_t * dst_pitch,
		  const void *src, const size_t * src_pitch,
		  const size_t * elem_number, size_t elem_size)
{
	assert(d > 0);
	size_t *target_dims;
	target_dims = (size_t *) alloca(d * sizeof(size_t));
	target_dims[0] = d - 1;
	for (size_t i = 1; i < d; i += 1)
		target_dims[i] = i - 1;
	aml_copy_shnd(d, target_dims, dst, dst_pitch, src, src_pitch,
		      elem_number, elem_size);
	return 0;
}

int aml_copy_rtnd_c(size_t d, void *dst, const size_t * cumul_dst_pitch,
		    const void *src, const size_t * cumul_src_pitch,
		    const size_t * elem_number, size_t elem_size)
{
	assert(d > 0);
	size_t *target_dims;
	target_dims = (size_t *) alloca(d * sizeof(size_t));
	target_dims[0] = d - 1;
	for (size_t i = 1; i < d; i += 1)
		target_dims[i] = i - 1;
	aml_copy_shnd_c(d, target_dims, dst, cumul_dst_pitch, src,
			cumul_src_pitch, elem_number, elem_size);
	return 0;
}

int aml_copy_tndstr(size_t d, void *dst, const size_t * dst_pitch,
		    const size_t * dst_stride, const void *src,
		    const size_t * src_pitch, const size_t * src_stride,
		    const size_t * elem_number, size_t elem_size)
{
	assert(d > 0);
	size_t *target_dims;
	target_dims = (size_t *) alloca(d * sizeof(size_t));
	target_dims[d - 1] = 0;
	for (size_t i = 0; i < d - 1; i += 1)
		target_dims[i] = i + 1;
	aml_copy_shndstr(d, target_dims, dst, dst_pitch, dst_stride, src,
			 src_pitch, src_stride, elem_number, elem_size);
	return 0;
}

int aml_copy_tndstr_c(size_t d, void *dst, const size_t * cumul_dst_pitch,
		      const size_t * dst_stride, const void *src,
		      const size_t * cumul_src_pitch, const size_t * src_stride,
		      const size_t * elem_number, size_t elem_size)
{
	assert(d > 0);
	size_t *target_dims;
	target_dims = (size_t *) alloca(d * sizeof(size_t));
	target_dims[d - 1] = 0;
	for (size_t i = 0; i < d - 1; i += 1)
		target_dims[i] = i + 1;
	aml_copy_shndstr_c(d, target_dims, dst, cumul_dst_pitch, dst_stride,
			   src, cumul_src_pitch, src_stride, elem_number,
			   elem_size);
	return 0;
}

int aml_copy_rtndstr(size_t d, void *dst, const size_t * dst_pitch,
		     const size_t * dst_stride, const void *src,
		     const size_t * src_pitch, const size_t * src_stride,
		     const size_t * elem_number, size_t elem_size)
{
	assert(d > 0);
	size_t *target_dims;
	target_dims = (size_t *) alloca(d * sizeof(size_t));
	target_dims[0] = d - 1;
	for (size_t i = 1; i < d; i += 1)
		target_dims[i] = i - 1;
	aml_copy_shndstr(d, target_dims, dst, dst_pitch, dst_stride, src,
			 src_pitch, src_stride, elem_number, elem_size);
	return 0;
}

int aml_copy_rtndstr_c(size_t d, void *dst, const size_t * cumul_dst_pitch,
		       const size_t * dst_stride, const void *src,
		       const size_t * cumul_src_pitch,
		       const size_t * src_stride, const size_t * elem_number,
		       size_t elem_size)
{
	assert(d > 0);
	size_t *target_dims;
	target_dims = (size_t *) alloca(d * sizeof(size_t));
	target_dims[0] = d - 1;
	for (size_t i = 1; i < d; i += 1)
		target_dims[i] = i - 1;
	aml_copy_shndstr_c(d, target_dims, dst, cumul_dst_pitch, dst_stride,
			   src, cumul_src_pitch, src_stride, elem_number,
			   elem_size);
	return 0;
}

int aml_copy_layout_native(struct aml_layout *dst, const struct aml_layout *src)
{
	size_t d;
	size_t elem_size;
	struct aml_layout_data *ddst;
	struct aml_layout_data *dsrc;
	ddst = dst->data;
	dsrc = src->data;
	d = dsrc->ndims;
	assert(d > 0);
	elem_size = dsrc->cpitch[0];
	assert(d == ddst->ndims);
	assert(elem_size == ddst->cpitch[0]);
	for (size_t i = 0; i < d; i += 1)
		assert(dsrc->dims[i] == ddst->dims[i]);
	return aml_copy_ndstr_c(d, ddst->ptr, ddst->cpitch, ddst->stride,
				dsrc->ptr, dsrc->cpitch, dsrc->stride,
				dsrc->dims, elem_size);
}

int aml_copy_layout_transform_native(struct aml_layout *dst,
				     const struct aml_layout *src,
				     const size_t * target_dims)
{
	size_t d;
	size_t elem_size;
	struct aml_layout_data *ddst;
	struct aml_layout_data *dsrc;
	ddst = dst->data;
	dsrc = src->data;
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

int aml_copy_layout_transpose_native(struct aml_layout *dst,
				     const struct aml_layout *src)
{
	size_t d;
	size_t *target_dims;
	d = src->data->ndims;
	target_dims = (size_t *) alloca(d * sizeof(size_t));
	target_dims[d - 1] = 0;
	for (size_t i = 0; i < d - 1; i += 1)
		target_dims[i] = i + 1;
	return aml_copy_layout_transform_native(dst, src, target_dims);
}

int aml_copy_layout_reverse_transpose_native(struct aml_layout *dst,
					     const struct aml_layout *src)
{
	size_t d;
	size_t *target_dims;
	d = src->data->ndims;
	target_dims = (size_t *) alloca(d * sizeof(size_t));
	target_dims[0] = d - 1;
	for (size_t i = 1; i < d; i += 1)
		target_dims[i] = i - 1;
	return aml_copy_layout_transform_native(dst, src, target_dims);
}

static inline void aml_copy_layout_generic_helper(size_t d,
						  struct aml_layout *dst,
						  const struct aml_layout *src,
						  const size_t * elem_number,
						  size_t elem_size,
						  size_t * coords)
{
	if (d == 1)
		for (size_t i = 0; i < elem_number[0]; i += 1) {
			coords[0] = i;
			coords[0] = i;
			memcpy(aml_layout_aderef(dst, coords),
			       aml_layout_aderef(src, coords), elem_size);
	} else
		for (size_t i = 0; i < elem_number[d - 1]; i += 1) {
			coords[d - 1] = i;
			coords[d - 1] = i;
			aml_copy_layout_generic_helper(d - 1, dst, src,
						       elem_number, elem_size,
						       coords);
		}
}

static inline void aml_copy_layout_transform_generic_helper(size_t d,
							    struct aml_layout
							    *dst,
							    const struct
							    aml_layout *src,
							    const size_t *
							    elem_number,
							    size_t elem_size,
							    size_t * coords,
							    size_t * coords_out,
							    const size_t *
							    target_dims)
{
	if (d == 1)
		for (size_t i = 0; i < elem_number[target_dims[0]]; i += 1) {
			coords_out[0] = i;
			coords[target_dims[0]] = i;
			memcpy(aml_layout_aderef(dst, coords_out),
			       aml_layout_aderef(src, coords), elem_size);
	} else
		for (size_t i = 0; i < elem_number[target_dims[d - 1]]; i += 1) {
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

int aml_copy_layout_generic(struct aml_layout *dst,
			    const struct aml_layout *src)
{
	size_t d;
	size_t elem_size;
	size_t *coords;
	size_t *elem_number;
	assert(aml_layout_ndims(dst) == aml_layout_ndims(src));
	d = aml_layout_ndims(dst);
	assert(aml_layout_element_size(dst) == aml_layout_element_size(src));
	elem_size = aml_layout_element_size(dst);
	coords = (size_t *) alloca(d * sizeof(size_t));
	elem_number = (size_t *) alloca(d * sizeof(size_t));
	aml_layout_adims(src, elem_number);
	aml_copy_layout_generic_helper(d, dst, src, elem_number, elem_size,
				       coords);
	return 0;
}

int aml_copy_layout_transform_generic(struct aml_layout *dst,
				      const struct aml_layout *src,
				      const size_t * target_dims)
{
	size_t d;
	size_t elem_size;
	size_t *coords;
	size_t *coords_out;
	size_t *elem_number;
	assert(aml_layout_ndims(dst) == aml_layout_ndims(src));
	d = aml_layout_ndims(dst);
	assert(aml_layout_element_size(dst) == aml_layout_element_size(src));
	elem_size = aml_layout_element_size(dst);
	coords = (size_t *) alloca(d * sizeof(size_t));
	coords_out = (size_t *) alloca(d * sizeof(size_t));
	elem_number = (size_t *) alloca(d * sizeof(size_t));
	aml_layout_adims(src, elem_number);
	aml_copy_layout_transform_generic_helper(d, dst, src, elem_number,
						 elem_size, coords, coords_out,
						 target_dims);
	return 0;
}

int aml_copy_layout_transpose_generic(struct aml_layout *dst,
				      const struct aml_layout *src)
{
	size_t d;
	size_t *target_dims;
	d = src->data->ndims;
	target_dims = (size_t *) alloca(d * sizeof(size_t));
	target_dims[d - 1] = 0;
	for (size_t i = 0; i < d - 1; i += 1)
		target_dims[i] = i + 1;
	return aml_copy_layout_transform_generic(dst, src, target_dims);
}

int aml_copy_layout_reverse_transpose_generic(struct aml_layout *dst,
					      const struct aml_layout *src)
{
	size_t d;
	size_t *target_dims;
	d = src->data->ndims;
	target_dims = (size_t *) alloca(d * sizeof(size_t));
	target_dims[0] = d - 1;
	for (size_t i = 1; i < d; i += 1)
		target_dims[i] = i - 1;
	return aml_copy_layout_transform_generic(dst, src, target_dims);
}
