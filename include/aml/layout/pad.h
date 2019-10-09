/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
*******************************************************************************/

#ifndef AML_LAYOUT_PAD_H
#define AML_LAYOUT_PAD_H 1

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

int aml_layout_pad_create(struct aml_layout **layout, const int order,
			  struct aml_layout *target, const size_t *dim,
			  void *neutral);

void aml_layout_pad_destroy(struct aml_layout **layout);

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

#endif
