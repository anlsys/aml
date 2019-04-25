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

/**
 * @defgroup aml_tiling_1d "AML 1D Tiling"
 * @brief 1 dimension tiling implementation.
 *
 * Implementation of 1D tilings.
 * @{
 **/

/** Initialized structure containing operations on 1D tiling. **/
extern struct aml_tiling_ops aml_tiling_1d_ops;

/** Initialized structure containing operations on 1D tiling. **/
extern struct aml_tiling_iterator_ops aml_tiling_iterator_1d_ops;

/**
 * Data of 1 dimension tiling. 1D tiling consists in a set of
 * contiguous data blocks.
 **/
struct aml_tiling_1d_data {
	/** The size of a data block in tiling **/
	size_t blocksize;
	/** The toal size of the tiling **/
	size_t totalsize;
};

/** Data of 1 dimension tiling iterator. **/
struct aml_tiling_iterator_1d_data {
	/** Index of the current iteration **/
	size_t i;
	/** Tiling beeing iterated **/
	struct aml_tiling_1d_data *tiling;
};

/** Static declaration of 1D tiling **/
#define AML_TILING_1D_DECL(name) \
	struct aml_tiling_1d_data __ ##name## _inner_data; \
	struct aml_tiling name = { \
		&aml_tiling_1d_ops, \
		(struct aml_tiling_data *)&__ ## name ## _inner_data, \
	}

/** Static declaration of 1D tiling iterator **/
#define AML_TILING_ITERATOR_1D_DECL(name) \
	struct aml_tiling_iterator_1d_data __ ##name## _inner_data; \
	struct aml_tiling_iterator name = { \
		&aml_tiling_iterator_1d_ops, \
		(struct aml_tiling_iterator_data *)&__ ## name ## _inner_data, \
	}

/** Static declaration of the size of a 1D tiling **/
#define AML_TILING_1D_ALLOCSIZE (sizeof(struct aml_tiling_1d_data) + \
				 sizeof(struct aml_tiling))

/** Static declaration of the size of a 1D tiling iterator **/
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

/**
 * @}
 **/

#endif /* AML_TILING_1D_H */
