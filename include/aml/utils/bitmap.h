#ifndef AML_BITMAP_H
#define AML_BITMAP_H

/*******************************************************************************
 * bitmap API:
 ******************************************************************************/

#define AML_BITMAP_TYPE        unsigned long
#define AML_BITMAP_NUM         8
#define AML_BITMAP_SIZE        (AML_BITMAP_NUM * sizeof(AML_BITMAP_TYPE))
#define AML_BITMAP_LEN         (8 * AML_BITMAP_SIZE)

typedef AML_BITMAP_TYPE* aml_bitmap;

/**
 * On stack allocation of new bitmap.
 **/
#define aml_bitmap_decl(name) AML_BITMAP_TYPE name[AML_BITMAP_NUM]

/**
 * On heap allocation of new empty bitmap.
 **/
aml_bitmap aml_bitmap_alloc();

/**
 * On heap allocation of a copy of an existing bitmap.
 **/
aml_bitmap aml_bitmap_dup(const aml_bitmap a);

/**
 * Free on heap allocated bitmap.
 **/
void aml_bitmap_free(aml_bitmap bitmap);

/**
 * Copy bitmap content.
 **/
void aml_bitmap_copy(aml_bitmap dst, const aml_bitmap src);

/**
 * Empty a bitmap with all bits cleared.
 **/
void aml_bitmap_zero(aml_bitmap bitmap);

/**
 * Fill a bitmap with all bits set.
 **/
void aml_bitmap_fill(aml_bitmap bitmap);

/**
 * Check whether a bit in bitmap is set.
 * Returns -1 if i is greater than bitmap length, 
 * 0 if bit is not set else a positive value..
 **/
int aml_bitmap_isset(const aml_bitmap bitmap, const unsigned i);

/**
 * Check whether a bit in bitmap is empty.
 * Returns 1 if yes, else 0.
 **/
int aml_bitmap_iszero(const aml_bitmap bitmap);

/**
 * Check whether a bit in bitmap is full.
 * Returns 1 if yes, else 0.
 **/
int aml_bitmap_isfull(const aml_bitmap bitmap);

/**
 * Check whether two bitmaps have identical value.
 **/
int aml_bitmap_isequal(const aml_bitmap a, const aml_bitmap b);

/**
 * Set a bit in bitmap.
 * Returns -1 if i is greater than bitmap length, else 0.
 **/
int aml_bitmap_set(aml_bitmap bitmap, const unsigned i);

/**
 * Clear a bit in bitmap.
 * Returns -1 if i is greater than bitmap length, else 0.
 **/
int aml_bitmap_clear(aml_bitmap bitmap, const unsigned i);

/**
 * Set a range [[i, ii]] of bits in bitmap.
 * Returns -1 if i or ii is greater than bitmap length, else 0.
 **/
int aml_bitmap_set_range(aml_bitmap bitmap, const unsigned i, const unsigned ii);

/**
 * Clear a range [[i, ii]] of bits in bitmap.
 * Returns -1 if i or ii is greater than bitmap length, else 0.
 **/
int aml_bitmap_clear_range(aml_bitmap bitmap, const unsigned i, const unsigned ii);
	
/**
 * Count the number of bits set in bitmap.
 **/
unsigned long aml_bitmap_nset(const aml_bitmap bitmap);

#endif //AML_BITMAP_H

