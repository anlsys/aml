/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
*******************************************************************************/

#ifndef AML_TILING_1D_H
#define AML_TILING_1D_H 1
/*******************************************************************************
 * Tiling 1D:
 ******************************************************************************/

extern struct aml_tiling_ops aml_tiling_1d_ops;
extern struct aml_tiling_iterator_ops aml_tiling_iterator_1d_ops;

struct aml_tiling_1d_data {
	size_t blocksize;
	size_t totalsize;
};

struct aml_tiling_iterator_1d_data {
	size_t i;
	struct aml_tiling_1d_data *tiling;
};

#define AML_TILING_1D_DECL(name) \
	struct aml_tiling_1d_data __ ##name## _inner_data; \
	struct aml_tiling name = { \
		&aml_tiling_1d_ops, \
		(struct aml_tiling_data *)&__ ## name ## _inner_data, \
	}

#define AML_TILING_ITERATOR_1D_DECL(name) \
	struct aml_tiling_iterator_1d_data __ ##name## _inner_data; \
	struct aml_tiling_iterator name = { \
		&aml_tiling_iterator_1d_ops, \
		(struct aml_tiling_iterator_data *)&__ ## name ## _inner_data, \
	}

#define AML_TILING_1D_ALLOCSIZE (sizeof(struct aml_tiling_1d_data) + \
				 sizeof(struct aml_tiling))

#define AML_TILING_ITERATOR_1D_ALLOCSIZE \
	(sizeof(struct aml_tiling_iterator_1d_data) + \
	 sizeof(struct aml_tiling_iterator))

/**
 * Allocates and initializes a new 1D tiling.
 *
 * @param tiling  an address where the pointer to the newly allocated tiling
 *           structure will be stored.
 * @param tilesize provides the size of each tile.
 * @param totalsize provides the size of the complete user data structure to be
 *   tiled.
 * @return 0 if successful; an error code otherwise.
 **/
int aml_tiling_1d_create(struct aml_tiling **tiling,
			 size_t tilesize, size_t totalsize);
/**
 * Initializes a tiling. Similar to create.
 *
 * @param tiling a tiling declared with AML_TILING_1D_DECL.
 * @param 0 if successful; an error code otherwise.
 **/
int aml_tiling_1d_init(struct aml_tiling *tiling, size_t tilesize,
		       size_t totalsize);

/**
 * Finalize a tiling.
 **/
void aml_tiling_1d_fini(struct aml_tiling *tiling);

/**
 * Tears down an initialized tiling.
 *
 * @param tiling a tiling created with aml_tiling_1d_create. NULL after return.
 **/
void aml_tiling_1d_destroy(struct aml_tiling **tiling);

#endif /* AML_TILING_1D_H */
