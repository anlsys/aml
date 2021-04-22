/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
*******************************************************************************/

#ifndef AML_TILING_PAD_H
#define AML_TILING_PAD_H 1

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup aml_tiling_pad "AML Padded Tiling"
 * @brief tiling with padding at the border
 *
 * Implementation of a tiling for which the border tiles are padded up to the
 * requested size.
 * @{
 **/

/** Initialized structure containing operations for a tiling in column order.**/
extern struct aml_tiling_ops aml_tiling_pad_column_ops;

/** Initialized structure containing operations for a tiling in row order. **/
extern struct aml_tiling_ops aml_tiling_pad_row_ops;

struct aml_tiling_pad {
	int tags;
	const struct aml_layout *layout;
	size_t ndims;
	size_t *tile_dims;
	size_t *dims;
	size_t *border_tile_dims;
	size_t *pad;
	void *neutral;
};

int aml_tiling_pad_create(struct aml_tiling **t, int tags,
			  const struct aml_layout *l, size_t ndims,
			  const size_t *tile_dims, void *neutral);

void aml_tiling_pad_destroy(struct aml_tiling **t);

/**
 * @}
 **/

#ifdef __cplusplus
}
#endif

#endif /* AML_TILING_PAD_H */
