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

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup aml_bitmap "AML Bitmap API"
 * @brief  AML Bitmap API
 *
 * AML defines its own bitmap and operations
 * on bitmap through this API.
 * @{
 **/

/** The maximum number of bits held in bitmap **/
#define AML_BITMAP_MAX   2048
/** The size in Bytes of aml_bitmap **/
#define AML_BITMAP_BYTES (AML_BITMAP_MAX/8)
/** The type used to store bits **/
#define AML_BITMAP_TYPE  unsigned long
/** The number of basic type elements used to store bits **/
#define AML_BITMAP_SIZE  ((int)(AML_BITMAP_BYTES/sizeof(AML_BITMAP_TYPE)))
/** The number of bits held in each basic type element **/
#define AML_BITMAP_NBITS ((int)(8 * sizeof(AML_BITMAP_TYPE)))

/**
 * aml_bitmap is a static array of elements wrapped in a structure.
 * aml_bitmap can be statically allocated.
 **/
struct aml_bitmap {
	unsigned long mask[AML_BITMAP_SIZE];
};

/**
 * Copy bitmap content.
 * @param dst: The bitmap where bits are copied.
 * @param src: The bitmap to copy.
 **/
void aml_bitmap_copy(struct aml_bitmap *dst, const struct aml_bitmap *src);

/**
 * Empty a bitmap with all bits cleared.
 * @param bitmap: The bitmap to set.
 **/
int aml_bitmap_zero(struct aml_bitmap *bitmap);

/**
 * Fill a bitmap with all bits set.
 * @param bitmap: The bitmap to set.
 **/
int aml_bitmap_fill(struct aml_bitmap *bitmap);

/**
 * Check whether a bit in bitmap is set.
 * @param bitmap: The bitmap to inspect.
 * @param i: the index of the bit to check.
 * @return -1 if i is greater than bitmap length.
 * @return 0 if bit is not set else a positive value.
 **/
int aml_bitmap_isset(const struct aml_bitmap *bitmap, const unsigned int i);

/**
 * Check whether a bit in bitmap is empty.
 * @param bitmap: The bitmap to inspect.
 * @return 1 if yes, else 0.
 **/
int aml_bitmap_iszero(const struct aml_bitmap *bitmap);

/**
 * Check whether a bit in bitmap is full.
 * @param bitmap: The bitmap to inspect.
 * @return 1 if yes, else 0.
 **/
int aml_bitmap_isfull(const struct aml_bitmap *bitmap);

/**
 * Check whether two bitmaps have identical value.
 * @param a: The left-hand side bitmap.
 * @param b: The right-hand side bitmap.
 * @return 1 if bitmaps match.
 * @return 0 if bitmaps do not match.
 **/
int aml_bitmap_isequal(const struct aml_bitmap *a, const struct aml_bitmap *b);

/**
 * Set a bit in bitmap.
 * @param bitmap: The bitmap to set.
 * @param i: The bit to set in bitmap.
 * @return -1 if "i" is greater than bitmap length.
 * @return 0
 **/
int aml_bitmap_set(struct aml_bitmap *bitmap, const unsigned int i);

/**
 * Clear a bit in bitmap.
 * @param bitmap: The bitmap to set.
 * @param i: The bit to clear in bitmap.
 * @return -1 if i is greater than bitmap length.
 * @return 0
 **/
int aml_bitmap_clear(struct aml_bitmap *bitmap, const unsigned int i);

/**
 * Set a range [[i, ii]] of bits in bitmap.
 * @param bitmap: The bitmap to set.
 * @param i: The index of the first lower bit to set.
 * @param ii: The index of the last lower bit to set.
 * @return -1 if i or ii is greater than bitmap length.
 * @return 0
 **/
int aml_bitmap_set_range(struct aml_bitmap *bitmap,
			 const unsigned int i, const unsigned int ii);

/**
 * Clear a range [[i, ii]] of bits in bitmap.
 * @param bitmap: The bitmap to set.
 * @param i: The index of the first lower bit to clear.
 * @param ii: The index of the last lower bit to clear.
 * @return -1 if "i" or "ii" is greater than bitmap length.
 * @return 0
 **/
int aml_bitmap_clear_range(struct aml_bitmap *bitmap,
			   const unsigned int i, const unsigned int ii);

/**
 * Count the number of bits set in bitmap.
 * @param bitmap: The bitmap to inspect.
 * @return The number of bits set in bitmap.
 **/
int aml_bitmap_nset(const struct aml_bitmap *bitmap);

/**
 * Copy a unsigned long array used as a bitmap into an actual bitmap.
 * @param bitmap: The bitmap to set.
 * @param src: An array of unsigned long storing bits to copy.
 *        Bits are copied as is, i.e in the same order.
 * @param size: The index of the right most bit set in "src".
 **/
void aml_bitmap_copy_from_ulong(struct aml_bitmap *bitmap,
				const unsigned long *src, size_t size);

/**
 * Copy a bitmap into an unsigned long array.
 * @param bitmap: The bitmap to copy.
 * @param dst: An array of unsigned long storing bits to write.
 *        Bits are copied as is, i.e in the same order.
 * @param size: The maximum number of bits in "src".
 **/
void aml_bitmap_copy_to_ulong(const struct aml_bitmap *bitmap,
			      unsigned long *dst, size_t size);

/**
 * Get index of the first bit set in bitmap.
 * @param bitmap: The bitmap to inspect.
 * @return -1 if no bit is set.
 * @return The index of last bit set in bitmap.
 **/
int aml_bitmap_last(const struct aml_bitmap *bitmap);

/**
 * Get index of the last bit set in bitmap.
 * @param bitmap: The bitmap to inspect.
 * @return -1 if no bit is set.
 * @return The index of first bit set in bitmap.
 **/
int aml_bitmap_first(const struct aml_bitmap *bitmap);

/**
 * String conversion to bitmap.
 * @param bitmap: The bitmap to set from a string.
 * @param bitmap_str: A string value among the following:
 * - string output from aml_bitmap_to_string(),
 * - "all", "fill", "full" to set all bits,
 * - NULL, "none", "zero", "empty" to clear all bits,
 * - an integer string to set one bit,
 * - a comma separated list of integers string of bits to set.
 * @return 0 on success.
 * @return -1 if "bitmap_str" is invalid.
 **/
int aml_bitmap_from_string(struct aml_bitmap *bitmap, const char *bitmap_str);

/**
 * Bitmap conversion to string. Output strings are 1,0 wrapped in brackets [].
 * @param bitmap: The bitmap to convert to string.
 * @return 0 if conversion went right, else -1.
 **/
char *aml_bitmap_to_string(const struct aml_bitmap *bitmap);

/**
 * Allocate a new empty (all zero) struct aml_bitmap.
 * @param[out] map pointer to an uninitialized struct aml_bitmap pointer to
 * receive the new bitmap.
 * @return On success, return 0 and map points to the new bitmap.
 * @return On failrure, sets map to NULL and returns one of the AML error codes:
 * -AML_ENOMEM if there wasn't enough memory available.
 **/
int aml_bitmap_create(struct aml_bitmap **map);

/**
 * Destroy (finalize and free resources) for a struct aml_bitmap created by
 * aml_bitmap_create.
 *
 * @param map: The a pointer to the bitmap to free. *map is NULL after
 *        this call.
 **/
void aml_bitmap_destroy(struct aml_bitmap **map);

/**
 * On heap allocation of a copy of an existing bitmap.
 * @param dst: A pointer to where the copy will be allocated.
 * @param src: The bitmap to duplicate.
 * @return 0 on success, -1 on error.
 **/
int aml_bitmap_dup(struct aml_bitmap **dst, const struct aml_bitmap *src);

/**
 * @}
 **/

#ifdef __cplusplus
}
#endif

#endif //AML_BITMAP_H

