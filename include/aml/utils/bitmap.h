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
 * On heap allocation of new empty bitmap.
 **/
struct aml_bitmap *aml_bitmap_create(void);

/**
 * On heap allocation of a copy of an existing bitmap.
 **/
struct aml_bitmap *aml_bitmap_dup(const struct aml_bitmap *src);

/**
 * Free on heap allocated bitmap.
 **/
void aml_bitmap_destroy(struct aml_bitmap *bitmap);

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
int aml_bitmap_isset(const struct aml_bitmap *bitmap, const unsigned i);

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
int aml_bitmap_set(struct aml_bitmap *bitmap, const unsigned i);

/**
 * Clear a bit in bitmap.
 * Returns -1 if i is greater than bitmap length, else 0.
 **/
int aml_bitmap_clear(struct aml_bitmap *bitmap, const unsigned i);

/**
 * Set a range [[i, ii]] of bits in bitmap.
 * Returns -1 if i or ii is greater than bitmap length, else 0.
 **/
int aml_bitmap_set_range(struct aml_bitmap *bitmap, const unsigned i, const unsigned ii);

/**
 * Clear a range [[i, ii]] of bits in bitmap.
 * Returns -1 if i or ii is greater than bitmap length, else 0.
 **/
int aml_bitmap_clear_range(struct aml_bitmap *bitmap, const unsigned i, const unsigned ii);
	
/**
 * Count the number of bits set in bitmap.
 **/
unsigned long aml_bitmap_nset(const struct aml_bitmap *bitmap);

/*
 * Copy a unsigned long array used as a bitmap into an actual bitmap.
 * Takes the array and its max set bit as input.
 */
void aml_bitmap_copy_ulong(struct aml_bitmap *bitmap, unsigned long *, size_t);

#endif //AML_BITMAP_H

