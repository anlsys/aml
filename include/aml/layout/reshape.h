/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
*******************************************************************************/

#ifndef AML_AREA_LAYOUT_RESHAPE_H
#define AML_AREA_LAYOUT_RESHAPE_H

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

int aml_layout_reshape_create(struct aml_layout **layout,
			      struct aml_layout *target,
			      const int order,
			      const size_t ndims,
			      const size_t *dims);

void aml_layout_reshape_destroy(struct aml_layout **l);

extern struct aml_layout_ops aml_layout_reshape_row_ops;
extern struct aml_layout_ops aml_layout_reshape_column_ops;

/**
 * @}
 **/

#endif
