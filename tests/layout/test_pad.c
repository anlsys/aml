/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
*******************************************************************************/

#include <assert.h>
#include "aml.h"
#include "aml/layout/dense.h"
#include "aml/layout/pad.h"

#include "test_layout.h"

void test_pad(int (*layout_create) (struct aml_layout **layout,
				    void *ptr,
				    const int order,
				    const size_t element_size,
				    const size_t ndims,
				    const size_t *dims,
				    const size_t *stride,
				    const size_t *pitch))
{
	struct aml_layout *a, *b;
	float memory[7][11];
	size_t dims[2] = { 7, 11 };
	size_t dims_pad[2] = { 11, 13 };
	float one = 1.0;
	size_t ret_dims[2];

	assert(!layout_create(&a, (void *)memory, AML_LAYOUT_ORDER_ROW_MAJOR,
			      sizeof(float), 2, dims, NULL, NULL));
	assert(!aml_layout_pad_create(&b, AML_LAYOUT_ORDER_ROW_MAJOR, a,
				      dims_pad, &one));

	assert(aml_layout_ndims(b) == 2);
	assert(!aml_layout_dims(b, ret_dims));
	assert(!memcmp(ret_dims, dims_pad, sizeof(size_t) * 2));
	ret_dims[0] = 10;
	ret_dims[1] = 12;
	assert(*(float *)aml_layout_deref(b, ret_dims) == one);
	test_layout_fprintf(stderr, "test-pad", b);
	test_layout_duplicate(b);
	aml_layout_destroy(&b);
}

int main(int argc, char *argv[])
{
	aml_init(&argc, &argv);
	test_pad(aml_layout_dense_create);
	aml_finalize();
	return 0;
}
