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
#include "aml/layout/dense.h"
#include "aml/dma/linux-spin.h"
#include <assert.h>

int main(int argc, char *argv[])
{
	struct aml_dma *dma;
	size_t isz = 1<<16;
	int idest[isz];
	int isrc[isz];
	struct aml_layout *idl, *isl;

	/* library initialization */
	aml_init(&argc, &argv);

	/* support data structures */
	assert(!aml_layout_dense_create(&idl, idest, 0, sizeof(int), 1, &isz,
					NULL, NULL));
	assert(!aml_layout_dense_create(&isl, isrc, 0, sizeof(int), 1, &isz,
					NULL, NULL));
	for (size_t i = 0; i < isz; i++) {
		idest[i] = 42;
		isrc[i] = 24;
	}
	/* invalid create input */
	assert(aml_dma_linux_spin_create(NULL, NULL, NULL) == -AML_EINVAL);

	/* invalid requests */
	assert(!aml_dma_linux_spin_create(&dma, NULL, NULL));
	assert(aml_dma_copy(dma, NULL, isl) == -AML_EINVAL);
	assert(aml_dma_copy(dma, idl, NULL) == -AML_EINVAL);

	struct aml_dma_request *r1, *r2;
	/* force dma to increase its requests queue */
	assert(!aml_dma_async_copy(dma, &r1, idl, isl));
	assert(!aml_dma_wait(dma, &r1));
	assert(!aml_dma_async_copy(dma, &r2, idl, isl));

	assert(aml_dma_wait(dma, NULL) == -AML_EINVAL);
	assert(!aml_dma_wait(dma, &r2));
	assert(!memcmp(isrc, idest, isz*sizeof(int)));
	aml_dma_linux_spin_destroy(&dma);

	/* destroy a running dma */
	assert(!aml_dma_linux_spin_create(&dma, NULL, NULL));
	assert(!aml_dma_async_copy(dma, &r1, idl, isl));
	aml_dma_linux_spin_destroy(&dma);

	/* delete everything */
	aml_layout_dense_destroy(&idl);
	aml_layout_dense_destroy(&isl);
	aml_finalize();
	return 0;
}
