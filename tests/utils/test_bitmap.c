/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include "aml.h"
#include <assert.h>

/* set of bits that are interesting to test. In increasing order by design, for
 * the range tests.
 * - first one
 * - middle of a backing element
 * - last one in the middle of the backing array
 * - middle of a backing element
 * - first one in the middle of the backing array
 * - middle of a backing element
 * - last one
 */
#define NTESTS 7
#define TESTS_IDX {0, AML_BITMAP_NBITS + 16, \
	(2*AML_BITMAP_NBITS) - 1, (2*AML_BITMAP_NBITS) + 16, \
	3*AML_BITMAP_NBITS, (3*AML_BITMAP_NBITS) + 16, AML_BITMAP_MAX - 1}
#define TESTS_OFF {0, 1, 1, 2, 3, 3, AML_BITMAP_SIZE - 1}

int main(void)
{
	struct aml_bitmap *bitmap;
	int idx[NTESTS] = TESTS_IDX;
	int off[NTESTS] = TESTS_OFF;

	/* create/destroy should fail on NULL */
	assert(aml_bitmap_create(NULL) == -AML_EINVAL);
	aml_bitmap_destroy(NULL);

	/* create should succeed with a pointer */
	assert(aml_bitmap_create(&bitmap) == AML_SUCCESS);
	assert(bitmap != NULL);

	/* access the mask just to be sure */
	assert(bitmap->mask[0] == 0);
	aml_bitmap_destroy(&bitmap);
	assert(bitmap == NULL);

	/* keep a bitmap on for the rest of the tests */
	assert(aml_bitmap_create(&bitmap) == AML_SUCCESS);

	/* zero should work */
	assert(aml_bitmap_zero(NULL) == -AML_EINVAL);
	memset(bitmap->mask, 255, AML_BITMAP_BYTES);
	assert(aml_bitmap_zero(bitmap) == AML_SUCCESS);
	for (size_t i = 0; i < AML_BITMAP_SIZE; i++)
		assert(bitmap->mask[i] == 0);

	/* iszero/isfull (1) */
	assert(aml_bitmap_iszero(NULL) == -AML_EINVAL);
	assert(aml_bitmap_isfull(NULL) == -AML_EINVAL);
	assert(aml_bitmap_iszero(bitmap));
	assert(!aml_bitmap_isfull(bitmap));

	/* nset (1) */
	assert(aml_bitmap_nset(bitmap) == 0);

	/* fill should work */
	assert(aml_bitmap_fill(NULL) == -AML_EINVAL);
	assert(aml_bitmap_fill(bitmap) == AML_SUCCESS);
	for (size_t i = 0; i < AML_BITMAP_SIZE; i++)
		assert(bitmap->mask[i] != 0);
	assert(!aml_bitmap_iszero(bitmap));
	assert(aml_bitmap_isfull(bitmap));

	/* nset (2) */
	assert(aml_bitmap_nset(bitmap) == AML_BITMAP_MAX);

	/* set / clear */
	aml_bitmap_zero(bitmap);
	assert(aml_bitmap_set(NULL, 0) == -AML_EINVAL);
	assert(aml_bitmap_set(NULL, AML_BITMAP_MAX) == -AML_EINVAL);
	for (int i = 0; i < NTESTS; i++) {
		aml_bitmap_zero(bitmap);
		assert(aml_bitmap_set(bitmap, idx[i]) == AML_SUCCESS);
		assert(bitmap->mask[off[i]] != 0);
		/* nset (3) */
		assert(aml_bitmap_nset(bitmap) == 1);
		/* clear that offset and check for zero */
		bitmap->mask[off[i]] = 0;
		assert(aml_bitmap_iszero(bitmap));
	}

	/* same logic for clear */
	assert(aml_bitmap_clear(NULL, 0) == -AML_EINVAL);
	assert(aml_bitmap_clear(NULL, AML_BITMAP_MAX) == -AML_EINVAL);
	for (int i = 0; i < NTESTS; i++) {
		aml_bitmap_fill(bitmap);
		assert(aml_bitmap_clear(bitmap, idx[i]) == AML_SUCCESS);
		assert(bitmap->mask[off[i]] != ~0UL);
		/* nset (4) */
		assert(aml_bitmap_nset(bitmap) == AML_BITMAP_MAX - 1);
		/* set that offset and check for full */
		bitmap->mask[off[i]] = ~0UL;
		assert(aml_bitmap_isfull(bitmap));
	}

	/* idem for isset */
	assert(aml_bitmap_isset(NULL, 0) == -AML_EINVAL);
	assert(aml_bitmap_isset(bitmap, AML_BITMAP_MAX) == -AML_EINVAL);
	aml_bitmap_zero(bitmap);
	for (int i = 1; i < NTESTS; i++) {
		aml_bitmap_set(bitmap, idx[i]);
		assert(aml_bitmap_isset(bitmap, idx[i]));
		assert(!aml_bitmap_isset(bitmap, idx[i-1]));
		aml_bitmap_clear(bitmap, idx[i]);
		assert(!aml_bitmap_isset(bitmap, idx[i]));
		assert(!aml_bitmap_isset(bitmap, idx[i-1]));
	}

	/* bad input for ranges (out-of-bounds, reverse indexes) */
	assert(aml_bitmap_set_range(NULL, 0, 1) == -AML_EINVAL);
	assert(aml_bitmap_set_range(bitmap, 0, AML_BITMAP_MAX) == -AML_EINVAL);
	assert(aml_bitmap_set_range(bitmap, AML_BITMAP_MAX,
				    AML_BITMAP_MAX+1) == -AML_EINVAL);
	assert(aml_bitmap_set_range(bitmap, AML_BITMAP_MAX, 0) == -AML_EINVAL);
	assert(aml_bitmap_set_range(bitmap, 1, 0) == -AML_EINVAL);

	for (int i = 0; i < NTESTS; i++) {
		for (int j = i; j < NTESTS; j++) {
			aml_bitmap_zero(bitmap);
			assert(aml_bitmap_set_range(bitmap, idx[i], idx[j]) ==
			       AML_SUCCESS);
			for (int k = 0; k < idx[i]; k++)
				assert(!aml_bitmap_isset(bitmap, k));
			for (int k = idx[i]; k <= idx[j]; k++)
				assert(aml_bitmap_isset(bitmap, k));
			for (int k = idx[j]+1; k < AML_BITMAP_MAX; k++)
				assert(!aml_bitmap_isset(bitmap, k));
			/* nset (5) */
			assert(aml_bitmap_nset(bitmap) == 1 + idx[j] - idx[i]);
		}
	}

	/* same logic for clear */
	assert(aml_bitmap_clear_range(NULL, 0, 1) == -AML_EINVAL);
	assert(aml_bitmap_clear_range(bitmap, 0, AML_BITMAP_MAX) ==
	       -AML_EINVAL);
	assert(aml_bitmap_clear_range(bitmap, AML_BITMAP_MAX,
				    AML_BITMAP_MAX+1) == -AML_EINVAL);
	assert(aml_bitmap_clear_range(bitmap, AML_BITMAP_MAX, 0) ==
	       -AML_EINVAL);
	assert(aml_bitmap_clear_range(bitmap, 1, 0) == -AML_EINVAL);

	for (int i = 0; i < NTESTS; i++) {
		for (int j = i; j < NTESTS; j++) {
			aml_bitmap_fill(bitmap);
			assert(aml_bitmap_clear_range(bitmap, idx[i], idx[j]) ==
			       AML_SUCCESS);
			for (int k = 0; k < idx[i]; k++)
				assert(aml_bitmap_isset(bitmap, k));
			for (int k = idx[i]; k <= idx[j]; k++)
				assert(!aml_bitmap_isset(bitmap, k));
			for (int k = idx[j]+1; k < AML_BITMAP_MAX; k++)
				assert(aml_bitmap_isset(bitmap, k));
			assert(aml_bitmap_nset(bitmap) ==
			       AML_BITMAP_MAX - idx[j] + idx[i] - 1);
		}
	}
	aml_bitmap_destroy(&bitmap);
	return 0;
}

