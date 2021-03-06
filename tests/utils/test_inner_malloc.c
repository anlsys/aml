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

int main(void)
{
	intptr_t *ptr = AML_INNER_MALLOC(void *, void *);

	assert(ptr != NULL);
	void *b = AML_INNER_MALLOC_GET_FIELD(ptr, 2, void *, void *);

	assert(b == &ptr[1]);
	free(ptr);
	return 0;
}

