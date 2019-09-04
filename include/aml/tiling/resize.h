/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
*******************************************************************************/

#ifndef AML_TILING_RESIZE_H
#define AML_TILING_RESIZE_H 1

/**
 * @defgroup aml_tiling_1d "AML Resizable Tiling"
 * @brief tiling with not homogeneous tiles
 *
 * Implementation of a tiling for which the border tiles have the exact size of
 * the underlying layout (not smaller, not larger).
 * @{
 **/

/** Initialized structure containing operations for a tiling in column order.
 **/
extern struct aml_tiling_ops aml_tiling_resize_column_ops;

/** Initialized structure containing operations for a tiling in row order. **/
extern struct aml_tiling_ops aml_tiling_resize_row_ops;

struct aml_tiling_resize {
	int tags;
	const struct aml_layout *layout;
	size_t ndims;
	size_t *tile_dims;
	size_t *dims;
	size_t *border_tile_dims;
};

int aml_tiling_resize_create(struct aml_tiling **t, int tags,
			     const struct aml_layout *l,
			     size_t ndims, const size_t *tile_dims);

int aml_tiling_resize_destroy(struct aml_tiling **t);

/**
 * @}
 **/

#endif /* AML_TILING_RESIZE_H */
