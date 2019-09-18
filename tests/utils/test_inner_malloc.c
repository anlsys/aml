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
#include <assert.h>

int main(void)
{
	intptr_t *ptr = AML_INNER_MALLOC_2(void *, void *);

	assert(ptr != NULL);
	void *b = AML_INNER_MALLOC_NEXTPTR(ptr, void *, void *);

	assert(b == &ptr[1]);
	free(ptr);
	return 0;
}

