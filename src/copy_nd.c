#include <aml.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <alloca.h>

static inline void aml_compute_cumulative_pitch(size_t d,
						size_t *cumul_dst_pitch,
						size_t *cumul_src_pitch,
						const size_t *dst_pitch,
						const size_t *src_pitch,
						size_t elem_size)
{
	cumul_dst_pitch[0] = elem_size;
	cumul_src_pitch[0] = elem_size;
	for (int i = 0; i < d - 1; i++) {
		cumul_dst_pitch[i + 1] = dst_pitch[i] * cumul_dst_pitch[i];
		cumul_src_pitch[i + 1] = src_pitch[i] * cumul_src_pitch[i];
	}
}

static inline void aml_copy_2d_helper(void *dst, const size_t *cumul_dst_pitch,
				      const void *src,
				      const size_t *cumul_src_pitch,
				      const size_t *elem_number,
				      size_t elem_size)
{
	if (cumul_dst_pitch[0] == elem_size && cumul_src_pitch[0] == elem_size)
		for (int i = 0; i < elem_number[1]; i++) {
			memcpy(dst, src, elem_number[0] * elem_size);
			dst = (void *)((uintptr_t) dst + cumul_dst_pitch[1]);
			src = (void *)((uintptr_t) src + cumul_src_pitch[1]);
		}
	else
		for (int j = 0; j < elem_number[1]; j++)
			for (int i = 0; i < elem_number[0]; i++)
				memcpy((void *)((uintptr_t) dst +
						i * cumul_dst_pitch[0] +
						j * cumul_dst_pitch[1]),
				       (void *)((uintptr_t) src +
						i * cumul_src_pitch[0] +
						j * cumul_src_pitch[1]),
				       elem_size);
}

static void aml_copy_nd_helper(size_t d, void *dst,
			       const size_t *cumul_dst_pitch, const void *src,
			       const size_t *cumul_src_pitch,
			       const size_t *elem_number,
			       const size_t elem_size)
{
	if (d == 1)
		if (cumul_dst_pitch[0] == elem_size &&
			cumul_src_pitch[0] == elem_size)
			memcpy(dst, src, elem_number[0] * elem_size);
		else
			for (int i = 0; i < elem_number[0]; i++)
				memcpy((void *)((uintptr_t) dst +
						i * cumul_dst_pitch[0]),
				       (void *)((uintptr_t) src +
						i * cumul_src_pitch[0]),
				       elem_size);
	else if (d == 2)
		aml_copy_2d_helper(dst, cumul_dst_pitch, src, cumul_src_pitch,
				   elem_number, elem_size);
	else {
		for (int i = 0; i < elem_number[d - 1]; i++) {
			aml_copy_nd_helper(d - 1, dst, cumul_dst_pitch, src,
					   cumul_src_pitch, elem_number,
					   elem_size);
			dst =
			    (void *)((uintptr_t) dst + cumul_dst_pitch[d - 1]);
			src =
			    (void *)((uintptr_t) src + cumul_src_pitch[d - 1]);
		}
	}
}

int aml_copy_nd_c(size_t d, void *dst, const size_t *cumul_dst_pitch,
		  const void *src, const size_t *cumul_src_pitch,
		  const size_t *elem_number, size_t elem_size)
{
	assert(d > 0);
	for (int i = 0; i < d - 1; i++) {
		assert(cumul_dst_pitch[i + 1] >= cumul_dst_pitch[i] *
						elem_number[i]);
		assert(cumul_src_pitch[i + 1] >= cumul_src_pitch[i] *
						elem_number[i]);
	}
	aml_copy_nd_helper(d, dst, cumul_dst_pitch, src, cumul_src_pitch,
			   elem_number, elem_size);
	return 0;
}

int aml_copy_nd(size_t d, void *dst, const size_t *dst_pitch, const void *src,
		const size_t *src_pitch, const size_t *elem_number,
		size_t elem_size)
{
	assert(d > 0);
	size_t *cumul_dst_pitch = (size_t *) alloca(d * sizeof(size_t));
	size_t *cumul_src_pitch = (size_t *) alloca(d * sizeof(size_t));

	aml_compute_cumulative_pitch(d, cumul_dst_pitch, cumul_src_pitch,
				     dst_pitch, src_pitch, elem_size);
	aml_copy_nd_c(d, dst, cumul_dst_pitch, src, cumul_src_pitch,
		      elem_number, elem_size);
	return 0;
}

static void aml_copy_ndstr_helper(size_t d, void *dst,
				  const size_t *cumul_dst_pitch,
				  const size_t *dst_stride, const void *src,
				  const size_t *cumul_src_pitch,
				  const size_t *src_stride,
				  const size_t *elem_number, size_t elem_size)
{
	if (d == 1)
		for (int i = 0; i < elem_number[0]; i++)
			memcpy((void *)((uintptr_t) dst +
					i * dst_stride[0] * cumul_dst_pitch[0]),
			       (void *)((uintptr_t) src +
					i * src_stride[0] * cumul_src_pitch[0]),
			       elem_size);
	else {
		for (int i = 0; i < elem_number[d - 1]; i++) {
			aml_copy_ndstr_helper(d - 1, dst, cumul_dst_pitch,
					      dst_stride, src, cumul_src_pitch,
					      src_stride, elem_number,
					      elem_size);
			dst =
			    (void *)((uintptr_t) dst +
				     cumul_dst_pitch[d - 1] * dst_stride[d -
									 1]);
			src =
			    (void *)((uintptr_t) src +
				     cumul_src_pitch[d - 1] * src_stride[d -
									 1]);
		}
	}
}

int aml_copy_ndstr_c(size_t d, void *dst, const size_t *cumul_dst_pitch,
		     const size_t *dst_stride, const void *src,
		   const size_t *cumul_src_pitch, const size_t *src_stride,
		   const size_t *elem_number, const size_t elem_size)
{
	assert(d > 0);
	for (int i = 0; i < d - 1; i++) {
		assert(cumul_dst_pitch[i + 1] >=
			   cumul_dst_pitch[i] *	elem_number[i] *
			       dst_stride[i]);
		assert(cumul_src_pitch[i + 1] >=
			   cumul_src_pitch[i] * elem_number[i] *
			       src_stride[i]);
	}
	aml_copy_ndstr_helper(d, dst, cumul_dst_pitch, dst_stride, src,
				      cumul_src_pitch, src_stride, elem_number,
				      elem_size);
	return 0;
}

int aml_copy_ndstr(size_t d, void *dst, const size_t *dst_pitch,
		   const size_t *dst_stride, const void *src,
		   const size_t *src_pitch, const size_t *src_stride,
		   const size_t *elem_number, const size_t elem_size)
{
	assert(d > 0);
	size_t *cumul_dst_pitch = (size_t *) alloca(d * sizeof(size_t));
	size_t *cumul_src_pitch = (size_t *) alloca(d * sizeof(size_t));

	aml_compute_cumulative_pitch(d, cumul_dst_pitch, cumul_src_pitch,
				     dst_pitch, src_pitch, elem_size);
	aml_copy_ndstr_c(d, dst, cumul_dst_pitch, dst_stride, src,
			      cumul_src_pitch, src_stride, elem_number,
			      elem_size);
	return 0;
}

static void aml_copy_sh2d_helper(const size_t *target_dims, void *dst,
				 const size_t *cumul_dst_pitch,
				 const void *src,
				 const size_t *cumul_src_pitch,
				 const size_t *elem_number,
				 const size_t elem_size)
{
	for (int j = 0; j < elem_number[1]; j++)
		for (int i = 0; i < elem_number[0]; i++)
			memcpy((void *)((uintptr_t) dst +
					i * cumul_dst_pitch[target_dims[0]] +
					j * cumul_dst_pitch[target_dims[1]]),
			       (void *)((uintptr_t) src +
					i * cumul_src_pitch[0] +
					j * cumul_src_pitch[1]), elem_size);
}

static void aml_copy_shnd_helper(size_t d, const size_t *target_dims,
				 void *dst, const size_t *cumul_dst_pitch,
				 const void *src,
				 const size_t *cumul_src_pitch,
				 const size_t *elem_number,
				 const size_t elem_size)
{
	if (d == 1)
		for (int i = 0; i < elem_number[0]; i++)
			memcpy((void *)((uintptr_t) dst +
					i * cumul_dst_pitch[target_dims[0]]),
			       (void *)((uintptr_t) src +
					i * cumul_src_pitch[0]), elem_size);
	if (d == 2)
		aml_copy_sh2d_helper(target_dims, dst, cumul_dst_pitch, src,
				     cumul_src_pitch, elem_number, elem_size);
	else {
		// process dimension d-1
		for (int i = 0; i < elem_number[d - 1]; i++) {
			aml_copy_shnd_helper(d - 1, target_dims, dst,
					     cumul_dst_pitch, src,
					     cumul_src_pitch, elem_number,
					     elem_size);
			dst =
			    (void *)((uintptr_t) dst +
				     cumul_dst_pitch[target_dims[d - 1]]);
			src =
			    (void *)((uintptr_t) src + cumul_src_pitch[d - 1]);
		}
	}
}

int aml_copy_shnd_c(size_t d, const size_t *target_dims, void *dst,
		    const size_t *cumul_dst_pitch, const void *src,
		    const size_t *cumul_src_pitch, const size_t *elem_number,
		    const size_t elem_size)
{
	assert(d > 0);
	size_t present_dims = 0;

	for (int i = 0; i < d; i++) {
		assert(target_dims[i] < d);
		if (target_dims[i] < d - 1)
			assert(cumul_dst_pitch[target_dims[i] + 1] >=
				   cumul_dst_pitch[target_dims[i]] *
				       elem_number[i]);
		present_dims |= 1 << target_dims[i];
	}
	for (int i = 0; i < d; i++)
		assert(present_dims & (1 << i));
	for (int i = 0; i < d - 1; i++)
		assert(cumul_src_pitch[i + 1] >= cumul_src_pitch[i] *
						     elem_number[i]);
	aml_copy_shnd_helper(d, target_dims, dst, cumul_dst_pitch, src,
			     cumul_src_pitch, elem_number, elem_size);
	return 0;
}

int aml_copy_shnd(size_t d, const size_t *target_dims, void *dst,
		  const size_t *dst_pitch, const void *src,
		  const size_t *src_pitch, const size_t *elem_number,
		  const size_t elem_size)
{
	assert(d > 0);
	size_t *cumul_dst_pitch = (size_t *) alloca(d * sizeof(size_t));
	size_t *cumul_src_pitch = (size_t *) alloca(d * sizeof(size_t));

	aml_compute_cumulative_pitch(d, cumul_dst_pitch, cumul_src_pitch,
				     dst_pitch, src_pitch, elem_size);
	aml_copy_shnd_c(d, target_dims, dst, cumul_dst_pitch, src,
			cumul_src_pitch, elem_number, elem_size);
	return 0;
}

int aml_copy_tnd(size_t d, void *dst, const size_t *dst_pitch, const void *src,
		 const size_t *src_pitch, const size_t *elem_number,
		 const size_t elem_size)
{
	assert(d > 0);
	size_t *target_dims = (size_t *) alloca(d * sizeof(size_t));

	target_dims[0] = d - 1;
	for (int i = 1; i < d; i++)
		target_dims[i] = i - 1;
	aml_copy_shnd(d, target_dims, dst, dst_pitch, src, src_pitch,
		      elem_number, elem_size);
	return 0;
}

int aml_copy_tnd_c(size_t d, void *dst, const size_t *cumul_dst_pitch,
		   const void *src, const size_t *cumul_src_pitch,
		   const size_t *elem_number, const size_t elem_size)
{
	assert(d > 0);
	size_t *target_dims = (size_t *) alloca(d * sizeof(size_t));

	target_dims[0] = d - 1;
	for (int i = 1; i < d; i++)
		target_dims[i] = i - 1;
	aml_copy_shnd_c(d, target_dims, dst, cumul_dst_pitch, src,
			cumul_src_pitch, elem_number, elem_size);
	return 0;
}

int aml_copy_rtnd(size_t d, void *dst, const size_t *dst_pitch,
		  const void *src, const size_t *src_pitch,
		  const size_t *elem_number, const size_t elem_size)
{
	assert(d > 0);
	size_t *target_dims = (size_t *) alloca(d * sizeof(size_t));

	target_dims[d - 1] = 0;
	for (int i = 0; i < d - 1; i++)
		target_dims[i] = i + 1;
	aml_copy_shnd(d, target_dims, dst, dst_pitch, src, src_pitch,
		      elem_number, elem_size);
	return 0;
}

int aml_copy_rtnd_c(size_t d, void *dst, const size_t *cumul_dst_pitch,
		    const void *src, const size_t *cumul_src_pitch,
		    const size_t *elem_number, const size_t elem_size)
{
	assert(d > 0);
	size_t *target_dims = (size_t *) alloca(d * sizeof(size_t));

	target_dims[d - 1] = 0;
	for (int i = 0; i < d - 1; i++)
		target_dims[i] = i + 1;
	aml_copy_shnd_c(d, target_dims, dst, cumul_dst_pitch, src,
			cumul_src_pitch, elem_number, elem_size);
	return 0;
}

static void aml_copy_shndstr_helper(size_t d, const size_t *target_dims,
				    void *dst, const size_t *cumul_dst_pitch,
				    const size_t *dst_stride, const void *src,
				    const size_t *cumul_src_pitch,
				    const size_t *src_stride,
				    const size_t *elem_number,
				    const size_t elem_size)
{
	if (d == 1)
		for (int i = 0; i < elem_number[0]; i++)
			memcpy((void *)((uintptr_t) dst +
					i * cumul_dst_pitch[target_dims[0]] *
					dst_stride[target_dims[0]]),
			       (void *)((uintptr_t) src +
					i * cumul_src_pitch[0] * src_stride[0]),
			       elem_size);
	else {
		// process dimension d-1
		for (int i = 0; i < elem_number[d - 1]; i++) {
			aml_copy_shndstr_helper(d - 1, target_dims, dst,
						cumul_dst_pitch, dst_stride,
						src, cumul_src_pitch,
						src_stride, elem_number,
						elem_size);
			dst =
			    (void *)((uintptr_t) dst +
				     cumul_dst_pitch[target_dims[d - 1]] *
				     dst_stride[target_dims[d - 1]]);
			src =
			    (void *)((uintptr_t) src +
				     cumul_src_pitch[d - 1] * src_stride[d -
									 1]);
		}
	}
}

int aml_copy_shndstr_c(size_t d, const size_t *target_dims, void *dst,
		       const size_t *cumul_dst_pitch, const size_t *dst_stride,
		       const void *src, const size_t *cumul_src_pitch,
		       const size_t *src_stride, const size_t *elem_number,
		       const size_t elem_size)
{
	assert(d > 0);
	size_t present_dims = 0;

	for (int i = 0; i < d; i++) {
		assert(target_dims[i] < d);
		if (target_dims[i] < d - 1)
			assert(cumul_dst_pitch[target_dims[i] + 1] >=
				   cumul_dst_pitch[target_dims[i]] *
				       elem_number[i] *
					   dst_stride[target_dims[i]]);
		present_dims |= 1 << target_dims[i];
	}
	for (int i = 0; i < d; i++)
		assert(present_dims & (1 << i));
	for (int i = 0; i < d - 1; i++)
		assert(cumul_src_pitch[i + 1] >= cumul_src_pitch[i] *
						     elem_number[i] *
							src_stride[i]);
	aml_copy_shndstr_helper(d, target_dims, dst, cumul_dst_pitch,
				dst_stride, src, cumul_src_pitch,
				src_stride, elem_number, elem_size);
	return 0;
}

int aml_copy_shndstr(size_t d, const size_t *target_dims, void *dst,
		     const size_t *dst_pitch, const size_t *dst_stride,
		     const void *src, const size_t *src_pitch,
		     const size_t *src_stride, const size_t *elem_number,
		     const size_t elem_size)
{
	assert(d > 0);
	size_t *cumul_dst_pitch = (size_t *) alloca(d * sizeof(size_t));
	size_t *cumul_src_pitch = (size_t *) alloca(d * sizeof(size_t));

	aml_compute_cumulative_pitch(d, cumul_dst_pitch, cumul_src_pitch,
				     dst_pitch, src_pitch, elem_size);
	aml_copy_shndstr_c(d, target_dims, dst, cumul_dst_pitch,
			   dst_stride, src, cumul_src_pitch,
			   src_stride, elem_number, elem_size);
	return 0;
}

int aml_copy_tndstr(size_t d, void *dst, const size_t *dst_pitch,
		    const size_t *dst_stride, const void *src,
		    const size_t *src_pitch, const size_t *src_stride,
		    const size_t *elem_number, const size_t elem_size)
{
	assert(d > 0);
	size_t *target_dims = (size_t *) alloca(d * sizeof(size_t));

	target_dims[0] = d - 1;
	for (int i = 1; i < d; i++)
		target_dims[i] = i - 1;
	aml_copy_shndstr(d, target_dims, dst, dst_pitch, dst_stride, src,
			 src_pitch, src_stride, elem_number, elem_size);
	return 0;
}

int aml_copy_tndstr_c(size_t d, void *dst, const size_t *cumul_dst_pitch,
		      const size_t *dst_stride, const void *src,
		      const size_t *cumul_src_pitch, const size_t *src_stride,
		      const size_t *elem_number, const size_t elem_size)
{
	assert(d > 0);
	size_t *target_dims = (size_t *) alloca(d * sizeof(size_t));

	target_dims[0] = d - 1;
	for (int i = 1; i < d; i++)
		target_dims[i] = i - 1;
	aml_copy_shndstr_c(d, target_dims, dst, cumul_dst_pitch, dst_stride,
			   src, cumul_src_pitch, src_stride, elem_number,
			   elem_size);
	return 0;
}

int aml_copy_rtndstr(size_t d, void *dst, const size_t *dst_pitch,
		     const size_t *dst_stride, const void *src,
		     const size_t *src_pitch, const size_t *src_stride,
		     const size_t *elem_number, const size_t elem_size)
{
	assert(d > 0);
	size_t *target_dims = (size_t *) alloca(d * sizeof(size_t));

	target_dims[d - 1] = 0;
	for (int i = 0; i < d - 1; i++)
		target_dims[i] = i + 1;
	aml_copy_shndstr(d, target_dims, dst, dst_pitch, dst_stride, src,
			 src_pitch, src_stride, elem_number, elem_size);
	return 0;
}

int aml_copy_rtndstr_c(size_t d, void *dst, const size_t *cumul_dst_pitch,
		       const size_t *dst_stride, const void *src,
		       const size_t *cumul_src_pitch, const size_t *src_stride,
		       const size_t *elem_number, const size_t elem_size)
{
	assert(d > 0);
	size_t *target_dims = (size_t *) alloca(d * sizeof(size_t));

	target_dims[d - 1] = 0;
	for (int i = 0; i < d - 1; i++)
		target_dims[i] = i + 1;
	aml_copy_shndstr_c(d, target_dims, dst, cumul_dst_pitch, dst_stride,
			   src, cumul_src_pitch, src_stride, elem_number,
			   elem_size);
	return 0;
}
