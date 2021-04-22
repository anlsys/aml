/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
*******************************************************************************/

#ifndef AML_LAYOUT_PAD_H
#define AML_LAYOUT_PAD_H 1

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup aml_layout_pad "AML Layout Pad"
 * @brief Padded layout.
 *
 * Padded layouts describe layouts that have been padded with neutral elements
 * along one or several of their dimensions.
 *
 * @code
 * #include <aml/layout/pad.h>
 * @endcode
 * @see aml_layout
 * @{
 **/

/**
 * Structure of a padded layout.
 **/
struct aml_layout_pad {
	/** tags for this layout **/
	int tags;
	/** underlying layout which shape is being extended. **/
	struct aml_layout *target;
	/** number of dimensions: same of underlying layout **/
	size_t ndims;
	/** size of an element **/
	size_t element_size;
	/** dimensions of the padded layout **/
	size_t *dims;
	/** dimensions of the underlying layout **/
	size_t *target_dims;
	/** pointer to a neutral element to use for padding **/
	void *neutral;
};

/**
 * Creates a padded layout on top of a target layout, and takes ownership of it
 * (destroy will allow destroy the target layout).
 * @param layout a pointer to where to store a newly allocated layout.
 * @param order the order in which dimensions are organized.
 * @param target targeted layout.
 * @param dims the number of elements along each dimension of the layout,
 * including targeted layout and its pad
 * @param neutral a pointer to a neutral element to fill the pad with (copied
 * internally).
 * @return -AML_ENOMEM if layout allocation failed
 * @return -AML_EINVAL if target, dims, or neutral are NULL
 * @return AML_SUCCESS if creation succeeded.
 **/
int aml_layout_pad_create(struct aml_layout **layout,
                          const int order,
                          struct aml_layout *target,
                          const size_t *dims,
                          void *neutral);

void aml_layout_pad_destroy(struct aml_layout *layout);

/**
 * Pre-existing operators for padded layout
 * with AML_LAYOUT_ORDER_COLUMN_MAJOR order.
 **/
extern struct aml_layout_ops aml_layout_pad_column_ops;

/**
 * Pre-existing operators for padded layout
 * with AML_LAYOUT_ORDER_COLUMN_MAJOR order.
 **/
extern struct aml_layout_ops aml_layout_pad_row_ops;

/**
 * @}
 **/

#ifdef __cplusplus
}
#endif

#endif
