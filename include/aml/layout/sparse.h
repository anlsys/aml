/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *******************************************************************************/

#ifndef AML_AREA_LAYOUT_SPARSE_H
#define AML_AREA_LAYOUT_SPARSE_H

#include <aml.h>

/**
 * @defgroup aml_layout_sparse "AML Layout Sparse"
 * @brief Default aml layout.
 *
 * Sparse layouts describe storage of a set of pointers.
 * This layout has only one dimension of n pointers.
 * Therefore, layout order does not matter. It is set to
 * AML_LAYOUT_ORDER_ROW_MAJOR. Each element have a different size but
 * aml_layout_element_size() will always return the size of a pointer. rawptr()
 * is not implemented since the layout contains several raw pointers. slice()
 *and reshape() functions are not implemented for this type of layout.
 *
 * @code
 * #include <aml/layout/sparse.h>
 * @endcode
 * @see aml_layout
 * @{
 **/

/**
 * The plain sparse layout structure.
 * Unlike dense layout, it does not have cartesian coordinates.
 * This layout is only a flat array of pointers.
 * It always has 1 dimension in row major order.
 * Its element size is the size of a pointer
 * It can be extended with a coordinate system (weather a dense layout
 * over pointers or a sparse ndimensional space associating a set of
 * coordinates to each pointer.
 * This is the minimal layout for performing dmas from/to set
 * of plain pointers.
 **/
struct aml_layout_sparse {
	/**
	 * Additional metadata for customizing this layout.
	 * Needed by cuda to add device number used by dma
	 * to target the right devices.
	 **/
	void *metadata;
	/**
	 * Size of metadata field used to duplicate this layout.
	 */
	size_t metadata_size;
	/** Number of pointers in the layout **/
	size_t nptr;
	/**
	 * Pointers of the layout (nptr elements)
	 **/
	void **ptrs;
	/**
	 * Pointers size (nptr elements)
	 * Pointers may hold data of different size.
	 * We need this information for dmas.
	 **/
	size_t *sizes;
};

/**
 * Sparse layout constructor.
 * Destroy with aml_layout_destroy() or free.
 * @param[out] layout: A pointer where to store a newly allocated layout.
 * @param[in] nptr: The number of pointers in the layout.
 * @param[in] ptrs: The pointer to the data structure described by this layout.
 * dimension. If NULL, pitch is set to the number of elements in each dimension.
 * @param[in] sizes: The size of memory area pointed to by each pointer.
 * @param[in] metadata: Extra data to embed in the layout.
 * If not NULL, it will be copied with memcpy.
 * @param[in] metadata_size: The size of metadata.
 * Must be 0 if no metadata is provided.
 * @return -AML_ENOMEM if layout allocation failed.
 * @return -AML_EINVAL if layout is NULL.
 * @return AML_SUCCESS if creation succeeded.
 * @see aml_layout_sparse
 **/
int aml_layout_sparse_create(struct aml_layout **layout,
                             const size_t nptr,
                             void **ptrs,
                             const size_t *sizes,
                             void *metadata,
                             const size_t metadata_size);

/**
 * Deref operator for sparse layout.
 * @param[in] data: Sparse layout structure.
 * @param[in] coords: An array of one coordinate. This coordinate is
 * the index in pointer array where to pick the element to return.
 * @see aml_layout_deref()
 * @see aml_layout_deref_native()
 **/
void *aml_layout_sparse_deref(const struct aml_layout_data *data,
                              const size_t *coords);

/**
 * @param[in] data: Sparse layout structure.
 * @return always AML_LAYOUT_ORDER_ROW_MAJOR.
 * @see aml_layout_order()
 **/
int aml_layout_sparse_order(const struct aml_layout_data *data);

/**
 * @param[in] data: Sparse layout structure.
 * @return always 1.
 * @see aml_layout_ndims()
 **/
size_t aml_layout_sparse_ndims(const struct aml_layout_data *data);

/**
 * @param[in] data: Sparse layout structure.
 * @return always sizeof(void*).
 * @see aml_layout_element_size()
 **/
size_t aml_layout_sparse_element_size(const struct aml_layout_data *data);

/**
 * Duplicate operator.
 * @param[in] layout: The input sparse layout.
 * @param[out] dest: A pointer where layout duplicate will be
 * allocated.
 * @return same as aml_layout_sparse_create().
 */
int aml_layout_sparse_duplicate(const struct aml_layout *layout,
                                struct aml_layout **dest);

/**
 * Pre-existing operators for sparse layout.
 * slice() and reshape() functions are not implemented for this type of layout.
 * rawptr() is not implemented since the layout contains several raw pointers.
 **/
extern struct aml_layout_ops aml_layout_sparse_ops;

/**
 * @}
 **/

#endif
