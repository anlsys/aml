/******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#ifndef AML_LAYOUT_CUDA_H
#define AML_LAYOUT_CUDA_H

#include <aml/layout/sparse.h>

/**
 * @defgroup aml_layout_cuda "AML Layout Cuda"
 * @brief Layout on device pointer.
 *
 * Cuda layout is a wrapper on other layout.
 * All operations are deferred to the embedded layout.
 * deref operation of the embedded layout is used to compute offset
 * on device pointer and return the appropriate offset.
 * Operations on this layout cannot be used on device side.
 * However the layout pointer (if it is a device pointer) can be used
 * on device side.
 *
 * @code
 * #include <aml/layout/cuda.h>
 * @endcode
 * @see aml_layout
 * @{
 **/

/** aml_layout data structure **/
struct aml_layout_cuda_data {
	/** Pointer to data on device. **/
	void *device_ptr;
	/** device id where ptr is located **/
	int device;
	/** user expected layout order **/
	int order;
	/** layout num dims **/
	size_t ndims;
	/** layout dims stored in row major order **/
	size_t *dims;
	/**
	 * Offset between elements of the same dimension.
	 * Offset is in number of elements.
	 **/
	size_t *stride;
	/**
	 * cumulative distances between two elements in the same
	 * dimension (pitch[0] is the element size in bytes).
	 **/
	size_t *cpitch;
};

/**
 * Create a new layout on device pointer with embedded layout.
 * @param[out] out: A pointer to receive the newly allocated layout.
 * @param[in] device_ptr: The pointer on which the layout has to work.
 * @param[in] device: The device id where the device_ptr is allocated.
 * @param[in] element_size: The size of elements in this layout.
 * @param[in] order: Order of dimensions in the layout.
 * @param[in] ndims: The number of dimensions in the layout.
 * @param[in] dims: The dimensions in the layout.
 * @param[in] stride: The empty -- in number of elements -- space between
 * consecutive elements of the same dimension, in number of elements.
 * @param[in] pitch: The space -- in number of element -- between 2 elements in
 * the next dimension.
 * @return AML_SUCCESS or -AML_ENOMEM if the memory allocation for layout
 * failed.
 **/
int aml_layout_cuda_create(struct aml_layout **out,
                           void *device_ptr,
                           int device,
                           const size_t element_size,
                           const int order,
                           const size_t ndims,
                           const size_t *dims,
                           const size_t *stride,
                           const size_t *pitch);

/**
 * Create a new sparse layout on a specific device with a set
 * of pointers allocated on this device.
 * Destroy with aml_layout_destroy() or free.
 * @see aml_layout_sparse_create()
 * @param[out] layout: A pointer where to store a newly allocated layout.
 * @param[in] nptr: The number of pointers in the layout.
 * @param[in] ptrs: The pointer to the data structure described by this layout.
 * dimension. If NULL, pitch is set to the number of elements in each dimension.
 * @param[in] sizes: The size of memory area pointed to by each pointer.
 * @param[in] device_id: The device number where pointers are allocated.
 * @return -AML_ENOMEM if layout allocation failed.
 * @return -AML_EINVAL if layout is NULL.
 * @return AML_SUCCESS if creation succeeded.
 * @see aml_layout_sparse
 */
inline int aml_layout_cuda_sparse_create(struct aml_layout **layout,
                                         const size_t nptr,
                                         void **ptrs,
                                         const size_t *sizes,
                                         int device_id)
{
	return aml_layout_sparse_create(layout, nptr, ptrs, sizes, &device_id,
	                                sizeof(int));
}

/** Always returns the pointer to device_ptr, whatever the coordinates. **/
void *aml_layout_cuda_deref(const struct aml_layout_data *data,
                            const size_t *coords);

/** Always returns the pointer to device_ptr, whatever the coordinates. **/
void *aml_layout_cuda_deref_native(const struct aml_layout_data *data,
                                   const size_t *coords);

/** Returns layout order **/
int aml_layout_cuda_order(const struct aml_layout_data *data);

/** Copies layout dims with user order. **/
int aml_layout_cuda_dims(const struct aml_layout_data *data, size_t *dims);

/** Copies layout dims in row major order. **/
int aml_layout_cuda_dims_native(const struct aml_layout_data *data,
                                size_t *dims);

/** Returns the number of dimensions in the layout. **/
size_t aml_layout_cuda_ndims(const struct aml_layout_data *data);

/** Returns the size of an element in the layout. **/
size_t aml_layout_cuda_element_size(const struct aml_layout_data *data);

/** Cuda layout operations **/
extern struct aml_layout_ops aml_layout_cuda_ops;

/**
 * @}
 **/

#endif // AML_LAYOUT_CUDA_H
