/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#ifndef AML_AREA_LAYOUT_NATIVE_H
#define AML_AREA_LAYOUT_NATIVE_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup aml_layout_native "AML Layout Internal API"
 * @brief Layout API for internal management of layouts.
 *
 * @code
 * #include <aml/layout/native.h>
 * @endcode
 * @{
 **/

/**
 * Function for derefencing elements of a layout inside the library.
 * Layout assumes data is always stored in AML_LAYOUT_ORDER_FORTRAN order.
 * Coordinates provided by the library will match the same order, i.e
 * last dimension first.
 * @param[in] layout: An initialized layout.
 * @param[in] coords: The coordinates on which to access data.
 * The first coordinate should be the last dimensions and so on to the last,
 * coordinate, last dimension.
 * @return A pointer to the dereferenced element on success.
 * @return NULL on failure with aml_errno set to the error reason.
 **/
void *aml_layout_deref_native(const struct aml_layout *layout,
			      const size_t *coords);

/**
 * Return the layout dimensions in the order they are actually stored
 * in the library.
 * @param[in] layout: An initialized layout.
 * @param[in] dims: The non-NULL array of dimensions to fill. It is
 * supposed to be large enough to contain ndims() elements.
 * @return AML_SUCCESS on success, else an AML error code.
 **/
int aml_layout_dims_native(const struct aml_layout *layout,
			   size_t *dims);


/**
 * Return a layout that is a subset of another layout.
 * The number of elements to subset along each dimension
 * must be compatible with offsets and strides.
 * This function checks that the amount of elements along
 * each dimensions of the slice actually fits in the original
 * layout.
 * @param[in] layout: An initialized layout.
 * @param[out] reshaped_layout: a pointer where to store a
 * newly allocated layout with the queried subset of the
 * original layout on succes.
 * @param[in] offsets: The index of the first element of the slice
 * in each dimension.
 * @param[in] dims: The number of elements of the slice along each
 * dimension .
 * @param[in] strides: The displacement (in number of elements) between
 * elements of the slice.
 * @return AML_SUCCESS on success, else an AML error code (<0).
 **/
int aml_layout_slice_native(const struct aml_layout *layout,
			    struct aml_layout **reshaped_layout,
			    const size_t *offsets,
			    const size_t *dims,
			    const size_t *strides);

/**
 * @}
 **/

#ifdef __cplusplus
}
#endif

#endif
