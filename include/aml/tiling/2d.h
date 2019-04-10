/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
*******************************************************************************/

#ifndef AML_TILING_2D_H
#define AML_TILING_2D_H 1

/*******************************************************************************
 * Tiling 2D:
 * a contiguous memory area composed of contiguous tiles arranged in 2D grid.
 ******************************************************************************/

extern struct aml_tiling_ops aml_tiling_2d_rowmajor_ops;
extern struct aml_tiling_ops aml_tiling_2d_colmajor_ops;
extern struct aml_tiling_iterator_ops aml_tiling_iterator_2d_ops;

struct aml_tiling_2d_data {
	size_t blocksize;
	size_t totalsize;
	size_t ndims[2]; /* # number of rows, # number of cols (in tiles) */
};

struct aml_tiling_iterator_2d_data {
	size_t i;
	struct aml_tiling_2d_data *tiling;
};

#define AML_TILING_2D_ROWMAJOR_DECL(name) \
	struct aml_tiling_2d_data __ ##name## _inner_data; \
	struct aml_tiling name = { \
		&aml_tiling_2d_rowmajor_ops, \
		(struct aml_tiling_data *)&__ ## name ## _inner_data, \
	}

#define AML_TILING_2D_COLMAJOR_DECL(name) \
	struct aml_tiling_2d_data __ ##name## _inner_data; \
	struct aml_tiling name = { \
		&aml_tiling_2d_colmajor_ops, \
		(struct aml_tiling_data *)&__ ## name ## _inner_data, \
	}

#define AML_TILING_ITERATOR_2D_DECL(name) \
	struct aml_tiling_iterator_2d_data __ ##name## _inner_data; \
	struct aml_tiling_iterator name = { \
		&aml_tiling_iterator_2d_ops, \
		(struct aml_tiling_iterator_data *)&__ ## name ## _inner_data, \
	}

#define AML_TILING_2D_ALLOCSIZE (sizeof(struct aml_tiling_2d_data) + \
				 sizeof(struct aml_tiling))

#define AML_TILING_ITERATOR_2D_ALLOCSIZE \
	(sizeof(struct aml_tiling_iterator_2d_data) + \
	 sizeof(struct aml_tiling_iterator))

/**
 * Allocates and initializes a new 2D tiling.
 *
 * @param tiling  an address where the pointer to the newly allocated tiling
 *           structure will be stored.
 * @param type a type of 2D tiling
 * @param tilesize provides the size of each tile.
 * @param totalsize provides the size of the complete user data structure to be
 *   tiled.
 * @param rowsize the number of tiles in a row
 * @param colsize the number of tiles in a column
 * @return 0 if successful; an error code otherwise.
 **/
int aml_tiling_2d_create(struct aml_tiling **tiling, int type,
			 size_t tilesize, size_t totalsize,
			 size_t rowsize, size_t colsize);
/**
 * Initializes a tiling. Similar to create.
 *
 * @param tiling a tiling declared with AML_TILING_2D_DECL.
 * @param 0 if successful; an error code otherwise.
 **/
int aml_tiling_2d_init(struct aml_tiling *tiling, int type,
		       size_t tilesize, size_t totalsize,
		       size_t rowsize, size_t colsize);

/**
 * Finalize a tiling.
 **/
void aml_tiling_2d_fini(struct aml_tiling *tiling);

/**
 * Tears down an initialized tiling.
 *
 * @param tiling a tiling created with aml_tiling_1d_create. NULL after return.
 **/
void aml_tiling_2d_destroy(struct aml_tiling **tiling);


#endif /* AML_TILING_2D_H */
