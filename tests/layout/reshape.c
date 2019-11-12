/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

/**
 * This file defines the functions to test on a supposedly dense layout.
 **/

#include "test_layout.h"
#include "aml/layout/reshape.h"

void test_layout_reshape(struct aml_layout *layout,
			 const size_t n_new_dims,
			 const size_t *new_dims)
{
	struct aml_layout *b, *c;

	assert(aml_layout_reshape(layout, &b, n_new_dims, new_dims) ==
	       AML_SUCCESS);
	assert(aml_layout_order(b) == aml_layout_order(layout));

	aml_layout_reshape_create(&c, layout, aml_layout_order(layout),
				  n_new_dims, new_dims);

	assert(aml_layout_isequal(b, c));

	free(b);
	free(c);
}
