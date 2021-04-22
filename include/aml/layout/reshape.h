/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#ifndef AML_AREA_LAYOUT_RESHAPE_H
#define AML_AREA_LAYOUT_RESHAPE_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup aml_layout_reshape "AML Layout Reshape"
 * @brief Default aml layout.
 *
 * Layout for reshaping dense layouts when reshape method
 * on dense layouts fails.
 *
 * @code
 * #include <aml/layout/reshape.h>
 * @endcode
 * @see aml_layout
 * @{
 **/

/**
 * Structure of reshape layout.
 **/
struct aml_layout_data_reshape {
	struct aml_layout *target;
	size_t ndims;
	size_t target_ndims;
	size_t *dims;
	size_t *coffsets;
	size_t *target_dims;
};

/**
 * Creates a reshaped layout on top of a target layout and takes ownership of it
 * (destroy will allow destroy the target layout).
 * @param layout a pointer to where to store the newly allocated layout
 * @param target targeted layout.
 * @param order the order in which dimensions are organized.
 * @param ndims the number of dimensions of the reshaped layout
 * @param dims the number of elements along each dimension of the reshaped
 * layout.
 * @return -AML_ENOMEM if layout allocation failed
 * @return -AML_EINVAL if target or dims are NULL
 * @return AML_SUCCESS if creation succeeded.
 **/
int aml_layout_reshape_create(struct aml_layout **layout,
			      struct aml_layout *target,
			      const int order,
			      const size_t ndims,
			      const size_t *dims);

void aml_layout_reshape_destroy(struct aml_layout *l);

extern struct aml_layout_ops aml_layout_reshape_row_ops;
extern struct aml_layout_ops aml_layout_reshape_column_ops;

/**
 * @}
 **/

#ifdef __cplusplus
}
#endif

#endif
