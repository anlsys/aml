/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#ifndef AML_AREA_LAYOUT_DENSE_H
#define AML_AREA_LAYOUT_DENSE_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup aml_layout_dense "AML Layout Dense"
 * @brief Default aml layout.
 *
 * Dense layouts describe how a multi-dimensional dense data structure
 * is collapsed into a linear (and contiguous) virtual address range.
 * Dimensions of a layout may contain a stride (space between elements)
 * on the virtual address space, and a pitch (distance between contiguous
 * elements of the same dimension).
 *
 * @code
 * #include <aml/layout/dense.h>
 * @endcode
 * @see aml_layout
 * @{
 **/

/**
 * Structure of a dense layout.
 **/
struct aml_layout_dense {
	/** base pointer of the address range **/
	void *ptr;
	/** number of dimensions **/
	size_t ndims;
	/**
	 * dimensions, in element size, of the data structure,
	 * by order of appearance in memory.
	 **/
	size_t *dims;
	/**
	 * Offset between elements of the same dimension.
	 * Offset in number of elements.
	 **/
	size_t *stride;
	/**
	 * cumulative distances between two elements in the same
	 * dimension (pitch[0] is the element size in bytes).
	 **/
	size_t *cpitch;
};

/**
 * Dense layout constructor.
 * @param[out] layout: A pointer where to store a newly allocated layout.
 * @param[in] ptr: The pointer to the data structure described by this layout.
 * @param[in] order: The order in which dimensions are organized.
 * Can be AML_LAYOUT_ORDER_COLUMN_MAJOR or AML_LAYOUT_ORDER_ROW_MAJOR.
 * @param[in] element_size: The size of each element in layout.
 * @param[in] ndims: The number of dimensions of the layout.
 * @param[in] dims: The number of elements along each dimension of the layout.
 * @param[in] stride: The space between elements (in number of elements),
 * along each dimension. If NULL then the stride is set to one for each
 * dimension.
 * @param[in] pitch: The space between consecutive elements of the same
 * dimension. If NULL, pitch is set to the number of elements in each dimension.
 * @return -AML_ENOMEM if layout allocation failed.
 * @return -AML_EINVAL if layout is NULL.
 * @return AML_SUCCESS if creation succeeded.
 * @see aml_layout_dense
 **/
int aml_layout_dense_create(struct aml_layout **layout,
			    void *ptr,
			    const int order,
			    const size_t element_size,
			    const size_t ndims,
			    const size_t *dims,
			    const size_t *stride,
			    const size_t *pitch);

/**
 * Deref operator for dense layout in AML_ORDER_COLUMN_MAJOR.
 * Also used as the deref operator for this type of layout.
 * Does not check its argument. If data is NULL, or coords are out
 * of bounds, the behaviour of aml_layout_column_deref() is undefined.
 * @see aml_layout_deref()
 * @see aml_layout_deref_native()
 **/
void *aml_layout_column_deref(const struct aml_layout_data *data,
			      const size_t *coords);

/**
 * Layout operator for retrieving order of dimension storage.
 * This function shall not fail.
 * @see aml_layout_order()
 **/
int aml_layout_column_order(const struct aml_layout_data *data);

/**
 * Operator for retrieving the number of dimensions in the layout.
 * Does not check data is not NULL. If data is NULL or not the good
 * pointer type, the behaviour is undefined.
 * @see aml_layout_ndims()
 **/
size_t aml_layout_dense_ndims(const struct aml_layout_data *data);

/**
 * Layout operator for retrieving layout elements size.
 * Does not check data is not NULL. If data is NULL or not the good
 * @see aml_layout_element_size()
 **/
size_t aml_layout_dense_element_size(const struct aml_layout_data *data);

/**
 * Operator for reshaping dense layouts with column major order.
 * Does not check if the number of elements match.
 * This should be done in aml_layout_reshape().
 * @return -AML_EINVAL if merge then split of dimensions
 * cannot be done appropriatly.
 * @return -AML_ENOMEM if the resulting layout cannot be allocated.
 * @return AML_SUCCESS on successful reshape.
 * @see aml_layout_reshape()
 **/
int aml_layout_column_reshape(const struct aml_layout_data *data,
			      struct aml_layout **output,
			      size_t ndims,
			      const size_t *dims);

/**
 * Operator for slicing dense layouts with column major order.
 * Does not check if slice elements are out of bound.
 * This should be done in aml_layout_slice().
 * @return -AML_ENOMEM if the resulting layout cannot be allocated.
 * @return AML_SUCCESS on successful slicing.
 * @see aml_layout_slice()
 **/
int aml_layout_column_slice(const struct aml_layout_data *data,
			    struct aml_layout **output,
			    const size_t *offsets,
			    const size_t *dims,
			    const size_t *strides);

/**
 * Operator for slicing dense layouts with column major order.
 * Does not check if slice elements are out of bound.
 * This should be done in aml_layout_slice().
 * @return -AML_ENOMEM if the resulting layout cannot be allocated.
 * @return AML_SUCCESS on successful slicing.
 * @see aml_layout_deref()
 **/
void *aml_layout_row_deref(const struct aml_layout_data *data,
			   const size_t *coords);

/**
 * Operator for retrieving layout order of a row major layout.
 * This function shall not fail.
 * @see aml_layout_order()
 **/
int aml_layout_row_order(const struct aml_layout_data *data);

/**
 * Operator for retrieving dimensions size of a layout with row major order.
 * Does not check data is not NULL. If data is NULL or not the good
 * pointer type, the behaviour is undefined.
 * Arguments are supposed to be checked in aml_layout_dims().
 * @see aml_layout_dims()
 **/
int aml_layout_row_dims(const struct aml_layout_data *data, size_t *dims);

/**
 * Operator for reshaping dense layouts with row major order.
 * Does not check if the number of elements match.
 * This should be done in aml_layout_reshape().
 * @return -AML_EINVAL if merge then split of dimensions
 * cannot be done appropriatly.
 * @return -AML_ENOMEM if the resulting layout cannot be allocated.
 * @return AML_SUCCESS on successful reshape.
 * @see aml_layout_reshape()
 **/
int aml_layout_row_reshape(const struct aml_layout_data *data,
			   struct aml_layout **output,
			   const size_t ndims,
			   const size_t *dims);

/**
 * Operator for slicing dense layouts with row major order.
 * Does not check if slice elements are out of bound.
 * This should be done in aml_layout_slice().
 * @return -AML_ENOMEM if the resulting layout cannot be allocated.
 * @return AML_SUCCESS on successful slicing.
 * @see aml_layout_slice()
 **/
int aml_layout_row_slice(const struct aml_layout_data *data,
			 struct aml_layout **output,
			 const size_t *offsets,
			 const size_t *dims,
			 const size_t *strides);

/**
 * Operator for slicing dense layouts with row major order,
 * without the overhead of user defined order, i.e using the internal
 * library order.
 * Does not check if slice elements are out of bound.
 * This should be done in aml_layout_slice().
 * @return -AML_ENOMEM if the resulting layout cannot be allocated.
 * @return AML_SUCCESS on successful slicing.
 * @see aml_layout_slice()
 **/
int aml_layout_row_slice_native(const struct aml_layout_data *data,
				struct aml_layout **output,
				const size_t *offsets,
				const size_t *dims,
				const size_t *strides);

/**
 * Pre-existing operators for dense layout
 * with AML_LAYOUT_ORDER_COLUMN_MAJOR order.
 **/
extern struct aml_layout_ops aml_layout_column_ops;

/**
 * Pre-existing operators for dense layout
 * with AML_LAYOUT_ORDER_ROW_MAJOR order.
 **/
extern struct aml_layout_ops aml_layout_row_ops;

/**
 * @}
 **/

#ifdef __cplusplus
}
#endif

#endif
