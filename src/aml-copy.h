#ifndef AML_COPY_H
#define AML_COPY_H 1

 /*******************************************************************************
 * Hypervolume copy and transpose functions.
 ******************************************************************************/

/*
 * Copies a (sub-)hypervolume to another (sub-)hypervolume.
 * "d": number of dimensions.
 * "dst": pointer to the destination hypervolume.
 * "dst_pitch": pointer to d-1 pitch values representing the pitch
 *              in each dimension of the destination hypervolume.
 * "src": pointer to the source hypervolume.
 * "src_pitch": pointer to d-1 pitch values representing the pitch
 *              in each dimension of the source hypervolume.
 * "elem_number": pointer to d values representing the number of elements
 *                in each dimension of the (sub-)hypervolume to copy.
 * "elem_size": size of memory elements.
 * Returns 0 if successful; an error code otherwise.
 */
int aml_copy_nd(size_t d, void *dst, const size_t *dst_pitch,
		const void *src, const size_t *src_pitch,
		const size_t *elem_number, const size_t elem_size);
/*
 * Copies a (sub-)hypervolume to another (sub-)hypervolume while transposing.
 * Reverse of aml_copy_rtnd.
 * Example a[3][4][5] -> b[5][3][4] (C notation).
 * "d": number of dimensions.
 * "dst": pointer to the destination hypervolume.
 * "dst_pitch": pointer to d-1 pitch values representing the pitch
 *              in each dimension of the destination hypervolume.
 * "src": pointer to the source hypervolume.
 * "src_pitch": pointer to d-1 pitch values representing the pitch
 *              in each dimension of the source hypervolume.
 * "elem_number": pointer to d values representing the number of elements
 *                in each dimension of the (sub-)hypervolume to copy.
 * "elem_size": size of memory elements in the src hypervolume order.
 * Returns 0 if successful; an error code otherwise.
 */
int aml_copy_tnd(size_t d, void *dst, const size_t *dst_pitch,
		 const void *src, const size_t *src_pitch,
		 const size_t *elem_number, const size_t elem_size);
/*
 * Copies a (sub-)hypervolume to another (sub-)hypervolume while transposing.
 * Reverse of aml_copy_tnd.
 * Example a[3][4][5] -> b[4][5][3] (C notation).
 * "d": number of dimensions.
 * "dst": pointer to the destination hypervolume.
 * "dst_pitch": pointer to d-1 pitch values representing the pitch
 *              in each dimension of the destination hypervolume.
 * "src": pointer to the source hypervolume.
 * "src_pitch": pointer to d-1 pitch values representing the pitch
 *              in each dimension of the source hypervolume.
 * "elem_number": pointer to d values representing the number of elements
 *                in each dimension of the (sub-)hypervolume to copy.
 * "elem_size": size of memory elements in the src hypervolume order.
 * Returns 0 if successful; an error code otherwise.
 */
int aml_copy_rtnd(size_t d, void *dst, const size_t *dst_pitch,
		  const void *src, const size_t *src_pitch,
		  const size_t *elem_number, const size_t elem_size);

/*
 * Copies a (sub-)hypervolume to another (sub-)hypervolume while shuffling
 * dimensions. Example a[4][2][3][5] -> b[5][4][3][2] (C notation).
 * "d": number of dimensions.
 * "target_dims": array of d dimension index representing the mapping
 *                between the source dimensions and the target dimensions.
 *                Example [3, 1, 0, 2]
 * "dst": pointer to the destination hypervolume.
 * "dst_pitch": pointer to d-1 pitch values representing the pitch
 *              in each dimension of the destination hypervolume.
 * "src": pointer to the source hypervolume.
 * "src_pitch": pointer to d-1 pitch values representing the pitch
 *              in each dimension of the source hypervolume.
 * "elem_number": pointer to d values representing the number of elements
 *                in each dimension of the (sub-)hypervolume to copy.
 * "elem_size": size of memory elements in the src hypervolume order.
 * Returns 0 if successful; an error code otherwise.
 */
int aml_copy_shnd(size_t d, const size_t *target_dims, void *dst,
		  const size_t *dst_pitch, const void *src,
		  const size_t *src_pitch, const size_t *elem_number,
		  const size_t elem_size);
/*
 * Strided version of aml_copy_nd.
 */
int aml_copy_ndstr(size_t d, void *dst, const size_t *dst_pitch,
		   const size_t *dst_stride, const void *src,
		   const size_t *src_pitch, const size_t *src_stride,
		   const size_t *elem_number, const size_t elem_size);
/*
 * Strided version of aml_copy_tnd.
 */
int aml_copy_tndstr(size_t d, void *dst, const size_t *dst_pitch,
		    const size_t *dst_stride, const void *src,
		    const size_t *src_pitch, const size_t *src_stride,
		    const size_t *elem_number, const size_t elem_size);
/*
 * Strided version of aml_copy_rtnd.
 */
int aml_copy_rtndstr(size_t d, void *dst, const size_t *dst_pitch,
		     const size_t *dst_stride, const void *src,
		     const size_t *src_pitch, const size_t *src_stride,
		     const size_t *elem_number, const size_t elem_size);
/*
 * Strided version of aml_copy_shnd.
 */
int aml_copy_shndstr(size_t d, const size_t *target_dims, void *dst,
		     const size_t *dst_pitch, const size_t *dst_stride,
		     const void *src, const size_t *src_pitch,
		     const size_t *src_stride, const size_t *elem_number,
		     const size_t elem_size);
/*
 * Version of aml_copy_nd using cumulative pitch.
 */
int aml_copy_nd_c(size_t d, void *dst, const size_t *cumul_dst_pitch,
		  const void *src, const size_t *cumul_src_pitch,
		  const size_t *elem_number, const size_t elem_size);
/*
 * Version of aml_copy_ndstr using cumulative pitch.
 */
int aml_copy_ndstr_c(size_t d, void *dst, const size_t *dst_pitch,
		     const size_t *cumul_dst_stride, const void *src,
		     const size_t *src_pitch, const size_t *cumul_src_stride,
		     const size_t *elem_number, const size_t elem_size);
/*
 * Version of aml_copy_nd using cumulative pitch.
 */
int aml_copy_tnd_c(size_t d, void *dst, const size_t *cumul_dst_pitch,
		   const void *src, const size_t *cumul_src_pitch,
		   const size_t *elem_number, const size_t elem_size);
/*
 * Version of aml_copy_nd using cumulative pitch.
 */
int aml_copy_rtnd_c(size_t d, void *dst, const size_t *cumul_dst_pitch,
		    const void *src, const size_t *cumul_src_pitch,
		    const size_t *elem_number, const size_t elem_size);
/*
 * Version of aml_copy_shnd using cumulative pitch.
 */
int aml_copy_shnd_c(size_t d, const size_t *target_dims, void *dst,
		    const size_t *cumul_dst_pitch, const void *src,
		    const size_t *cumul_src_pitch, const size_t *elem_number,
		    const size_t elem_size);
/*
 * Version of aml_copy_tndstr using cumulative pitch.
 */
int aml_copy_tndstr_c(size_t d, void *dst, const size_t *cumul_dst_pitch,
		      const size_t *dst_stride, const void *src,
		      const size_t *cumul_src_pitch, const size_t *src_stride,
		      const size_t *elem_number, const size_t elem_size);
/*
 * Version of aml_copy_rtndstr using cumulative pitch.
 */
int aml_copy_rtndstr_c(size_t d, void *dst, const size_t *cumul_dst_pitch,
		       const size_t *dst_stride, const void *src,
		       const size_t *cumul_src_pitch, const size_t *src_stride,
		       const size_t *elem_number, const size_t elem_size);
/*
 * Version of aml_copy_shndstr using cumulative pitch.
 */
int aml_copy_shndstr_c(size_t d, const size_t *target_dims, void *dst,
		       const size_t *cumul_dst_pitch, const size_t *dst_stride,
		       const void *src, const size_t *cumul_src_pitch,
		       const size_t *src_stride, const size_t *elem_number,
		       const size_t elem_size);

 /*******************************************************************************
 * Generic building block API: Native version
 * Native means using AML-internal layouts.
 ******************************************************************************/

int aml_copy_layout_native(struct aml_layout *dst,
			   const struct aml_layout *src);
int aml_copy_layout_transform_native(struct aml_layout *dst,
				     const struct aml_layout *src,
				     const size_t *target_dims);
int aml_copy_layout_generic(struct aml_layout *dst,
			    const struct aml_layout *src);
int aml_copy_layout_transform_generic(struct aml_layout *dst,
				      const struct aml_layout *src,
				      const size_t *target_dims);
int aml_copy_layout_transpose_native(struct aml_layout *dst, const struct aml_layout *src);
int aml_copy_layout_reverse_transpose_native(struct aml_layout *dst,
					     const struct aml_layout *src);
int aml_copy_layout_transpose_generic(struct aml_layout *dst, const struct aml_layout *src);
int aml_copy_layout_reverse_transpose_generic(struct aml_layout *dst,
					      const struct aml_layout *src);

#endif
