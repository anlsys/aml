#ifndef AML_BITMAP_HWLOC_H
#define AML_BITMAP_HWLOC_H

#include <hwloc.h>

/**
 * Translate from hwloc_bitmap_t to aml_bitmap.
 * Returns -1 if ab is NULL, else the last bit set if hb does not overflow ab, else
 * the first overflowing bit index.
 **/
int aml_bitmap_copy_hwloc_bitmap(struct aml_bitmap *ab, const hwloc_bitmap_t hb);

/**
 * Translate from aml_bitmap to hwloc_bitmap_t.
 * Returns -1 if ab is NULL, else the last bit set.
 **/
int hwloc_bitmap_copy_aml_bitmap(hwloc_bitmap_t hb, const struct aml_bitmap *ab);

/**
 * Allocate an hwloc_bitmap_t and translate from aml_bitmap to hwloc_bitmap_t.
 **/
hwloc_bitmap_t hwloc_bitmap_from_aml_bitmap(const struct aml_bitmap *b);

/**
 * Allocate an aml_bitmap and translate from hwloc_bitmap_t to aml_bitmap
 * If b is longer than aml_bitmap storage, b is truncated.
 **/
struct aml_bitmap * aml_bitmap_from_hwloc_bitmap(const hwloc_bitmap_t b);

#endif
