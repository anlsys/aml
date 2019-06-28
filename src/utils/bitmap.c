/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
*******************************************************************************/

#include "aml.h"
#include <string.h>

#define AML_BITMAP_EMPTY       (0UL)
#define AML_BITMAP_FULL        (~0UL)
#define AML_BITMAP_NTH(i)      ((i) / AML_BITMAP_NBITS)
#define AML_BITMAP_ITH(i)      (((i) % AML_BITMAP_NBITS))

/*******************************************************************************
 * General operators
*******************************************************************************/

void aml_bitmap_copy(struct aml_bitmap *dst, const struct aml_bitmap *src)
{
	if (dst == NULL || src == NULL)
		return;
	memcpy(dst, src, sizeof(struct aml_bitmap));
}

void aml_bitmap_copy_from_ulong(struct aml_bitmap *dst,
				const unsigned long *src, size_t maxbit)
{
	if (dst == NULL || src == NULL)
		return;
	if (maxbit > AML_BITMAP_MAX)
		maxbit = AML_BITMAP_MAX;
	for (size_t i = 0; i < maxbit; i++)
		if ((src[AML_BITMAP_NTH(i)] & (1UL << AML_BITMAP_ITH(i))) != 0)
			aml_bitmap_set(dst, i);
}

void aml_bitmap_copy_to_ulong(const struct aml_bitmap *dst,
			      unsigned long *src, size_t maxbit)
{
	if (dst == NULL || src == NULL)
		return;
	if (maxbit > AML_BITMAP_MAX)
		maxbit = AML_BITMAP_MAX;
	for (size_t i = 0; i < maxbit; i++)
		if (aml_bitmap_isset(dst, i))
			src[AML_BITMAP_NTH(i)] |= (1UL << AML_BITMAP_ITH(i));
}

void aml_bitmap_zero(struct aml_bitmap *bitmap)
{
	memset(bitmap, 0, sizeof(struct aml_bitmap));
}

int aml_bitmap_iszero(const struct aml_bitmap *bitmap)
{
	for (unsigned int i = 0; i < AML_BITMAP_SIZE; i++)
		if (bitmap->mask[i] != AML_BITMAP_EMPTY)
			return 0;
	return 1;
}

int aml_bitmap_isfull(const struct aml_bitmap *bitmap)
{
	for (unsigned int i = 0; i < AML_BITMAP_SIZE; i++)
		if (bitmap->mask[i] != AML_BITMAP_FULL)
			return 0;
	return 1;
}

void aml_bitmap_fill(struct aml_bitmap *bitmap)
{
	memset(bitmap, ~0, sizeof(struct aml_bitmap));
}

int aml_bitmap_isset(const struct aml_bitmap *bitmap, const unsigned int i)
{
	if (i >= AML_BITMAP_MAX)
		return -1;
	return (bitmap->mask[AML_BITMAP_NTH(i)] &
		(1UL << AML_BITMAP_ITH(i))) > 0UL;
}

int aml_bitmap_set(struct aml_bitmap *bitmap, const unsigned int i)
{
	if (i >= AML_BITMAP_MAX)
		return -1;
	bitmap->mask[AML_BITMAP_NTH(i)] |= (1UL << AML_BITMAP_ITH(i));
	return 0;
}

int aml_bitmap_isequal(const struct aml_bitmap *a, const struct aml_bitmap *b)
{
	for (unsigned int i = 0; i < AML_BITMAP_SIZE; i++)
		if (a->mask[i] != b->mask[i])
			return 0;
	return 1;
}

int aml_bitmap_clear(struct aml_bitmap *bitmap, const unsigned int i)
{
	if (i >= AML_BITMAP_MAX)
		return -1;
	bitmap->mask[AML_BITMAP_NTH(i)] &= ~(1UL << AML_BITMAP_ITH(i));
	return 0;
}

int aml_bitmap_set_range(struct aml_bitmap *bitmap,
			 const unsigned int i, const unsigned int ii)
{
	if (i >= AML_BITMAP_MAX || ii >= AML_BITMAP_MAX || i > ii)
		return -1;
	if (i == ii)
		return aml_bitmap_set(bitmap, i);

	unsigned long k = AML_BITMAP_ITH(ii + 1);
	unsigned long low = (AML_BITMAP_FULL << AML_BITMAP_ITH(i));
	unsigned long n = AML_BITMAP_NTH(i);
	unsigned long nn = AML_BITMAP_NTH(ii);
	unsigned long high = k == 0 ? AML_BITMAP_FULL : ~(AML_BITMAP_FULL << k);

	if (nn > n) {
		for (k = n + 1; k <= nn - 1; k++)
			bitmap->mask[k] = AML_BITMAP_FULL;
		bitmap->mask[n] |= low;
		bitmap->mask[nn] |= high;
	} else
		bitmap->mask[n] |= (low & high);

	return 0;
}

int aml_bitmap_clear_range(struct aml_bitmap *bitmap,
			   const unsigned int i, const unsigned int ii)
{
	if (i >= AML_BITMAP_MAX || ii >= AML_BITMAP_MAX || i > ii)
		return -1;
	if (i == ii)
		return aml_bitmap_clear(bitmap, i);

	unsigned long k = AML_BITMAP_ITH(ii + 1);
	unsigned long low = ~(AML_BITMAP_FULL << AML_BITMAP_ITH(i));
	unsigned long n = AML_BITMAP_NTH(i);
	unsigned long nn = AML_BITMAP_NTH(ii);
	unsigned long high = k == 0 ? AML_BITMAP_EMPTY : (AML_BITMAP_FULL << k);

	if (nn > n) {
		for (k = n + 1; k <= nn - 1; k++)
			bitmap->mask[k] = AML_BITMAP_EMPTY;
		bitmap->mask[n] &= low;
		bitmap->mask[nn] &= high;
	} else
		bitmap->mask[n] &= (low | high);

	return 0;
}

unsigned long aml_bitmap_nset(const struct aml_bitmap *bitmap)
{
	unsigned long i, b, n;
	unsigned long test = 1UL;
	unsigned long nset = 0;

	for (n = 0; n < AML_BITMAP_SIZE; n++) {
		b = bitmap->mask[n];
		for (i = 0; i < AML_BITMAP_NBITS; i++) {
			nset += b & test ? 1 : 0;
			b = b >> 1;
		}
	}
	return nset;
}

int aml_bitmap_last(const struct aml_bitmap *bitmap)
{
	if (bitmap == NULL)
		return -1;
	int n;
	unsigned int i = 0;

	for (n = AML_BITMAP_SIZE - 1; n >= 0 && bitmap->mask[n] == 0; n--)
		;

	if (n < 0)
		return -1;

	AML_BITMAP_TYPE mask = bitmap->mask[n];

	for (i = 0; i < AML_BITMAP_NBITS && mask; i++)
		mask = mask >> 1;

	return (AML_BITMAP_NBITS * n) + i - 1;
}

int aml_bitmap_first(const struct aml_bitmap *bitmap)
{
	if (bitmap == NULL)
		return -1;

	unsigned int n, i = 0;

	for (n = 0; n < AML_BITMAP_SIZE && bitmap->mask[n] == 0; n++)
		;

	if (n == AML_BITMAP_SIZE)
		return -1;

	AML_BITMAP_TYPE mask = bitmap->mask[n];

	for (i = 0; i < AML_BITMAP_NBITS && mask; i++)
		mask = mask << 1;

	int res = (AML_BITMAP_NBITS * n) + AML_BITMAP_NBITS - i;
	return res;

}

char *aml_bitmap_to_string(const struct aml_bitmap *bitmap)
{
	size_t i, len = AML_BITMAP_MAX + 3;
	char *output = malloc(len);

	if (output == NULL)
		return NULL;
	memset(output, 0, len);
	for (i = 1; i <= AML_BITMAP_MAX; i++)
		output[i] = aml_bitmap_isset(bitmap, (int)(i - 1)) ? '1' : '0';
	output[0] = '[';
	output[AML_BITMAP_MAX + 1] = ']';
	return output;
}

int aml_bitmap_from_string(struct aml_bitmap *bitmap, const char *bitmap_str)
{
	if (bitmap_str == NULL ||
	    !strcasecmp(bitmap_str, "none") ||
	    !strcasecmp(bitmap_str, "zero") ||
	    !strcasecmp(bitmap_str, "empty")) {
		aml_bitmap_zero(bitmap);
		return 0;
	}
	if (!strcasecmp(bitmap_str, "all") ||
	    !strcasecmp(bitmap_str, "fill") ||
	    !strcasecmp(bitmap_str, "full")) {
		aml_bitmap_fill(bitmap);
		return 0;
	}

	struct aml_bitmap b;
	size_t i;

	aml_bitmap_zero(&b);
	if (bitmap_str[0] == '[') {
		if (strlen(bitmap_str) < AML_BITMAP_MAX + 2 ||
		    bitmap_str[AML_BITMAP_MAX + 1] != ']')
			return -1;

		for (i = 1; i <= AML_BITMAP_MAX; i++) {
			if (bitmap_str[i] == '1')
				aml_bitmap_set(&b, i - 1);
			else if (bitmap_str[i] != '0')
				return -1;
		}
	} else if (bitmap_str[0] == '0' ||
		   bitmap_str[0] == '1' ||
		   bitmap_str[0] == '2' ||
		   bitmap_str[0] == '3' ||
		   bitmap_str[0] == '4' ||
		   bitmap_str[0] == '5' ||
		   bitmap_str[0] == '6' ||
		   bitmap_str[0] == '7' ||
		   bitmap_str[0] == '8' || bitmap_str[0] == '9') {
		char *saveptr, *tok, *str = strdup(bitmap_str);
		int bit;

		tok = strtok_r(str, ",", &saveptr);
		while (tok != NULL) {
			bit = atoi(tok);
			if (bit < 0 || bit >= AML_BITMAP_MAX) {
				free(str);
				return -1;
			}
			aml_bitmap_set(&b, bit);
			tok = strtok_r(NULL, ",", &saveptr);
		}
		free(str);
	} else {
		return -1;
	}

	aml_bitmap_copy(bitmap, &b);
	return 0;
}

/*******************************************************************************
 * create/destroy and others
*******************************************************************************/

int aml_bitmap_create(struct aml_bitmap **map)
{
	struct aml_bitmap *b = malloc(sizeof(struct aml_bitmap));

	if (b == NULL) {
		*map = NULL;
		return -AML_ENOMEM;
	}
	aml_bitmap_zero(b);
	*map = b;
	return 0;
}

/**
 * Initialize (zero a struct aml_bitmap). Not necessary on stack allocated
 * bitmaps.
 * @return 0 on success (always).
 **/
int aml_bitmap_init(struct aml_bitmap *map)
{
	aml_bitmap_zero(map);
	return 0;
}

/**
 * Finalize a struct aml_bitmap. This is an empty function.
 **/
void aml_bitmap_fini(__attribute__ ((unused)) struct aml_bitmap *map)
{
}

/**
 * Destroy (finalize and free resources) for a struct aml_bitmap created by
 * aml_bitmap_create.
 *
 * @param map is NULL after this call.
 **/
void aml_bitmap_destroy(struct aml_bitmap **map)
{
	free(*map);
	*map = NULL;
}

/**
 * On heap allocation of a copy of an existing bitmap.
 **/
int aml_bitmap_dup(struct aml_bitmap **dst, const struct aml_bitmap *src)
{
	struct aml_bitmap *ret;
	int err;

	err = aml_bitmap_create(&ret);
	if (err) {
		*dst = NULL;
		return err;
	}
	aml_bitmap_copy(ret, src);
	*dst = ret;
	return 0;
}
