/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
*******************************************************************************/

#ifndef AML_BITMAP_H
#define AML_BITMAP_H

/*******************************************************************************
 * bitmap API:
 ******************************************************************************/

#define AML_BITMAP_MAX   2048
#define AML_BITMAP_BYTES (AML_BITMAP_MAX/8)
#define AML_BITMAP_TYPE  unsigned long
#define AML_BITMAP_SIZE  (AML_BITMAP_BYTES/sizeof(AML_BITMAP_TYPE))
#define AML_BITMAP_NBITS (8 * sizeof(AML_BITMAP_TYPE))

struct aml_bitmap {
	unsigned long mask[AML_BITMAP_SIZE];
};

/**
 * Copy bitmap content.
 **/
void aml_bitmap_copy(struct aml_bitmap *dst, const struct aml_bitmap *src);

/**
 * Empty a bitmap with all bits cleared.
 **/
void aml_bitmap_zero(struct aml_bitmap *bitmap);

/**
 * Fill a bitmap with all bits set.
 **/
void aml_bitmap_fill(struct aml_bitmap *bitmap);

/**
 * Check whether a bit in bitmap is set.
 * Returns -1 if i is greater than bitmap length,
 * 0 if bit is not set else a positive value..
 **/
int aml_bitmap_isset(const struct aml_bitmap *bitmap, const unsigned int i);

/**
 * Check whether a bit in bitmap is empty.
 * Returns 1 if yes, else 0.
 **/
int aml_bitmap_iszero(const struct aml_bitmap *bitmap);

/**
 * Check whether a bit in bitmap is full.
 * Returns 1 if yes, else 0.
 **/
int aml_bitmap_isfull(const struct aml_bitmap *bitmap);

/**
 * Check whether two bitmaps have identical value.
 **/
int aml_bitmap_isequal(const struct aml_bitmap *a, const struct aml_bitmap *b);

/**
 * Set a bit in bitmap.
 * Returns -1 if i is greater than bitmap length, else 0.
 **/
int aml_bitmap_set(struct aml_bitmap *bitmap, const unsigned int i);

/**
 * Clear a bit in bitmap.
 * Returns -1 if i is greater than bitmap length, else 0.
 **/
int aml_bitmap_clear(struct aml_bitmap *bitmap, const unsigned int i);

/**
 * Set a range [[i, ii]] of bits in bitmap.
 * Returns -1 if i or ii is greater than bitmap length, else 0.
 **/
int aml_bitmap_set_range(struct aml_bitmap *bitmap,
			 const unsigned int i, const unsigned int ii);

/**
 * Clear a range [[i, ii]] of bits in bitmap.
 * Returns -1 if i or ii is greater than bitmap length, else 0.
 **/
int aml_bitmap_clear_range(struct aml_bitmap *bitmap,
			   const unsigned int i, const unsigned int ii);

/**
 * Count the number of bits set in bitmap.
 **/
unsigned long aml_bitmap_nset(const struct aml_bitmap *bitmap);

/**
 * Copy a unsigned long array used as a bitmap into an actual bitmap.
 * Takes the array and its max set bit as input.
 **/
void aml_bitmap_copy_from_ulong(struct aml_bitmap *bitmap,
				const unsigned long *src, size_t size);

/**
 * Copy a bitmap into an unsigned long array.
 * Takes the array and its max set bit as input.
 **/
void aml_bitmap_copy_to_ulong(const struct aml_bitmap *bitmap,
			      unsigned long *dst, size_t size);

/**
 * Get index of the first bit set in bitmap.
 * Return -1 if no bit is set.
 **/
int aml_bitmap_last(const struct aml_bitmap *bitmap);

/**
 * Get index of the last bit set in bitmap.
 * Return -1 if no bit is set.
 **/
int aml_bitmap_first(const struct aml_bitmap *bitmap);

/**
 * String conversion to bitmap.
 * Input accepts string output from aml_bitmap_to_string() or special values:
 * - "all", "fill", "full" to set all bits,
 * - NULL, "none", "zero", "empty" to clear all bits,
 * - an integer string to set one bit,
 * - a comma separated list of integers string of bits to set.
 **/
int aml_bitmap_from_string(struct aml_bitmap *bitmap, const char *bitmap_str);

/**
 * Bitmap conversion to string. Output strings are 1,0 wrapped in brackets [].
 * Returns 0 if conversion went right, else -1.
 **/
char *aml_bitmap_to_string(const struct aml_bitmap *bitmap);

/*******************************************************************************
 * create/destroy and others
*******************************************************************************/

/* Not needed, here for consistency */
#define AML_BITMAP_DECL(name) struct aml_bitmap name
#define AML_BITMAP_ALLOCSIZE (sizeof(struct aml_bitmap))

/**
 * Allocate a new empty (all zero) struct aml_bitmap.
 * @param[out] map pointer to an uninitialized struct aml_bitmap pointer to
 * receive the new bitmap.
 * @return On success, return 0 and map points to the new bitmap.
 * @return On failrure, sets map to NULL and returns one of the AML error codes:
 * - AML_ENOMEM if there wasn't enough memory available.
 **/
int aml_bitmap_create(struct aml_bitmap **map);

/**
 * Initialize (zero a struct aml_bitmap). Not necessary on stack allocated
 * bitmaps.
 * @return 0 on success (always).
 **/
int aml_bitmap_init(struct aml_bitmap *map);

/**
 * Finalize a struct aml_bitmap. This is an empty function.
 **/
void aml_bitmap_fini(struct aml_bitmap *map);

/**
 * Destroy (finalize and free resources) for a struct aml_bitmap created by
 * aml_bitmap_create.
 *
 * @param map is NULL after this call.
 **/
void aml_bitmap_destroy(struct aml_bitmap **map);

/**
 * On heap allocation of a copy of an existing bitmap.
 **/
int aml_bitmap_dup(struct aml_bitmap **dst, const struct aml_bitmap *src);

#endif //AML_BITMAP_H

