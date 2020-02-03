/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
*******************************************************************************/

#include <assert.h>
#include "test_layout.h"
#include "aml/layout/dense.h"

void test_dense(void)
{
	struct aml_layout *a, *b;

	/* padd the dims to the closest multiple of 2 */
	float memory[16][12][8][8][4];
	size_t cpitch[5] = {
		4,
		4 * 4,
		4 * 4 * 8,
		4 * 4 * 8 * 8,
		4 * 4 * 8 * 8 * 12
	};
	size_t dims[5] = { 2, 3, 7, 11, 13 };
	size_t stride[5] = { 1, 2, 1, 1, 1 };

	size_t dims_col[5] = { 2, 3, 7, 11, 13 };
	size_t dims_row[5] = { 13, 11, 7, 3, 2 };

	size_t pitch_col[5] = { 4, 8, 8, 12, 16 };
	size_t pitch_row[5] = { 16, 12, 8, 8, 4 };

	size_t stride_col[5] = { 1, 2, 1, 1, 1 };
	size_t stride_row[5] = { 1, 1, 1, 2, 1 };

	for (size_t i = 0; i < 4 * 8 * 8 * 12 * 16; i++)
		((float *)(&memory[0][0][0][0][0]))[i] = (float)i;

	/* test invalid input */
	assert(aml_layout_dense_create(NULL, (void *)memory,
				       AML_LAYOUT_ORDER_COLUMN_MAJOR,
				       sizeof(int), 5, dims_col, stride_col,
				       pitch_col) == -AML_EINVAL);
	assert(aml_layout_dense_create(&a, NULL,
				       AML_LAYOUT_ORDER_COLUMN_MAJOR,
				       sizeof(int), 5, dims_col, stride_col,
				       pitch_col) == -AML_EINVAL);
	/* missing: we don't test the tags/order value */
	assert(aml_layout_dense_create(&a, (void *)memory,
				       AML_LAYOUT_ORDER_COLUMN_MAJOR,
				       0, 5, dims_col, stride_col,
				       pitch_col) == -AML_EINVAL);
	assert(aml_layout_dense_create(&a, (void *)memory,
				       AML_LAYOUT_ORDER_COLUMN_MAJOR,
				       sizeof(int), 0, dims_col, stride_col,
				       pitch_col) == -AML_EINVAL);
	assert(aml_layout_dense_create(&a, (void *)memory,
				       AML_LAYOUT_ORDER_COLUMN_MAJOR,
				       sizeof(int), 5, NULL, stride_col,
				       pitch_col) == -AML_EINVAL);
	aml_layout_dense_destroy(NULL);

	/* test partial data */
	assert(aml_layout_dense_create(&a,
				       (void *)memory,
				       AML_LAYOUT_ORDER_COLUMN_MAJOR,
				       sizeof(int),
				       5,
				       dims_col,
				       NULL, pitch_col) == AML_SUCCESS);
	aml_layout_dense_destroy(&a);
	assert(aml_layout_dense_create(&a,
				       (void *)memory,
				       AML_LAYOUT_ORDER_COLUMN_MAJOR,
				       sizeof(int),
				       5,
				       dims_col,
				       stride_col, NULL) == AML_SUCCESS);
	aml_layout_dense_destroy(&a);

	/* initialize column order layouts */
	assert(aml_layout_dense_create(&a,
				       (void *)memory,
				       AML_LAYOUT_ORDER_COLUMN_MAJOR,
				       sizeof(int),
				       5,
				       dims_col,
				       stride_col, pitch_col) == AML_SUCCESS);
	test_layout_base(a);

	assert(aml_layout_dense_create(&b,
				       (void *)memory,
				       AML_LAYOUT_ORDER_COLUMN_MAJOR,
				       sizeof(int),
				       5,
				       dims_col,
				       stride_col, pitch_col) == AML_SUCCESS);
	test_layout_base(b);

	struct aml_layout_dense *adataptr;
	struct aml_layout_dense *bdataptr;

	adataptr = (struct aml_layout_dense *)a->data;
	bdataptr = (struct aml_layout_dense *)b->data;
	assert((intptr_t) (adataptr->stride) - (intptr_t) (adataptr->dims)
	       == 5 * sizeof(size_t));

	/* some simple checks */
	assert(!memcmp(adataptr->dims, dims, sizeof(size_t) * 5));
	assert(!memcmp(adataptr->stride, stride, sizeof(size_t) * 5));
	assert(!memcmp(adataptr->cpitch, cpitch, sizeof(size_t) * 5));
	assert(!memcmp(bdataptr->dims, dims, sizeof(size_t) * 5));

	/* test column major subroutines */
	size_t dims_res[5];
	size_t coords_test_col[5] = { 1, 2, 3, 4, 5 };
	void *test_addr;
	void *res_addr = (void *)&memory[5][4][3][2 * 2][1];

	aml_layout_dims(a, dims_res);
	assert(!memcmp(dims_res, dims_col, sizeof(size_t) * 5));
	test_addr = aml_layout_deref(a, coords_test_col);
	assert(res_addr == test_addr);
	assert(aml_layout_order(a) == AML_LAYOUT_ORDER_COLUMN_MAJOR);

	aml_layout_dense_destroy(&a);

	/* test partial data */
	assert(aml_layout_dense_create(&a,
				       (void *)memory,
				       AML_LAYOUT_ORDER_ROW_MAJOR,
				       sizeof(float),
				       5, dims_row,
				       NULL, pitch_row) == AML_SUCCESS);
	aml_layout_dense_destroy(&a);
	assert(aml_layout_dense_create(&a, (void *)memory,
				       AML_LAYOUT_ORDER_ROW_MAJOR,
				       sizeof(float),
				       5, dims_row,
				       stride_row, NULL) == AML_SUCCESS);
	aml_layout_dense_destroy(&a);

	/* initialize row order layouts */
	assert(aml_layout_dense_create(&a, (void *)memory,
				       AML_LAYOUT_ORDER_ROW_MAJOR,
				       sizeof(float),
				       5, dims_row,
				       stride_row, pitch_row) == AML_SUCCESS);

	adataptr = (struct aml_layout_dense *)a->data;
	bdataptr = (struct aml_layout_dense *)b->data;
	assert((intptr_t) (adataptr->stride) - (intptr_t) (adataptr->dims)
	       == 5 * sizeof(size_t));

	/* some simple checks */
	assert(!memcmp(adataptr->dims, dims, sizeof(size_t) * 5));
	assert(!memcmp(adataptr->stride, stride, sizeof(size_t) * 5));
	assert(!memcmp(bdataptr->dims, dims, sizeof(size_t) * 5));
	assert(!memcmp(bdataptr->cpitch, cpitch, sizeof(size_t) * 5));

	/* test row major subroutines */
	size_t coords_test_row[5] = { 5, 4, 3, 2, 1 };

	aml_layout_dims(a, dims_res);
	assert(!memcmp(dims_res, dims_row, sizeof(size_t) * 5));
	test_addr = aml_layout_deref(a, coords_test_row);
	assert(res_addr == test_addr);
	assert(aml_layout_order(a) == AML_LAYOUT_ORDER_ROW_MAJOR);

	aml_layout_dense_destroy(&a);
	aml_layout_dense_destroy(&b);
}

void test_generics(void)
{
	size_t x = 4, y = 5, z = 6;
	uint64_t memory[x * 2 * y * 1 * z * 2];
	size_t stride[3] = { 2, 1, 2 };

	size_t dims_col[3] = { x, y, z };
	size_t pitch_col[3] = { 8, 5, 12 };
	size_t new_dims_col[2] = { x * y, z };

	size_t dims_row[3] = { z, y, x };
	size_t new_dims_row[2] = { z * y, x };

	struct aml_layout *layout;

	// Test slice and reshape layout with stride and pitch.
	assert(aml_layout_dense_create
	       (&layout, (void *)memory, AML_LAYOUT_ORDER_COLUMN_MAJOR,
		sizeof(*memory), 3, dims_col, stride,
		pitch_col) == AML_SUCCESS);
	test_slice_dense(layout);
	test_layout_reshape(layout, 2, new_dims_col);
	test_layout_fprintf(stderr, "test-dense", layout);
	free(layout);

	// Test slice and reshape layout row major.
	assert(aml_layout_dense_create
	       (&layout, (void *)memory, AML_LAYOUT_ORDER_ROW_MAJOR,
		sizeof(*memory), 3, dims_row, NULL, NULL) == AML_SUCCESS);
	test_slice_dense(layout);
	test_layout_reshape(layout, 2, new_dims_row);
	test_layout_fprintf(stderr, "test-dense", layout);
	free(layout);
}

int main(int argc, char *argv[])
{
	aml_init(&argc, &argv);
	test_dense();
	test_generics();
	aml_finalize();
	return 0;
}
