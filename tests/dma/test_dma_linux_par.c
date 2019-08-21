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
#include "aml/dma/linux-par.h"
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
	assert(aml_dma_linux_par_create(NULL, 1) == -AML_EINVAL);

	/* invalid requests */
	assert(!aml_dma_linux_par_create(&dma, 1));
	assert(aml_dma_copy(dma, 42) == -AML_EINVAL);
	assert(aml_dma_copy(dma, AML_DMA_REQUEST_TYPE_PTR, NULL, isrc, isz) ==
	       -AML_EINVAL);
	assert(aml_dma_copy(dma, AML_DMA_REQUEST_TYPE_PTR, idest, NULL, isz) ==
	       -AML_EINVAL);
	assert(aml_dma_copy(dma, AML_DMA_REQUEST_TYPE_PTR, idest, isrc, 0) ==
	       -AML_EINVAL);
	assert(aml_dma_copy(dma, AML_DMA_REQUEST_TYPE_LAYOUT, NULL, isl) ==
	       -AML_EINVAL);
	assert(aml_dma_copy(dma, AML_DMA_REQUEST_TYPE_LAYOUT, idl, NULL) ==
	       -AML_EINVAL);

	struct aml_dma_request *r1, *r2;
	/* force dma to increase its requests queue */
	assert(!aml_dma_async_copy(dma, &r1, AML_DMA_REQUEST_TYPE_LAYOUT,
				   idl, isl));
	assert(!aml_dma_async_copy(dma, &r2, AML_DMA_REQUEST_TYPE_LAYOUT,
				   idl, isl));

	assert(aml_dma_wait(dma, NULL) == -AML_EINVAL);
	assert(!aml_dma_wait(dma, &r1));
	assert(!aml_dma_wait(dma, &r2));
	aml_dma_linux_par_destroy(&dma);

	/* cancel a request on the fly */
	assert(!aml_dma_linux_par_create(&dma, 1));
	assert(aml_dma_cancel(dma, NULL) == -AML_EINVAL);
	assert(!aml_dma_async_copy(dma, &r1, AML_DMA_REQUEST_TYPE_LAYOUT,
				   idl, isl));
	assert(!aml_dma_cancel(dma, &r1));

	/* destroy a running dma */
	assert(!aml_dma_async_copy(dma, &r1, AML_DMA_REQUEST_TYPE_LAYOUT,
				   idl, isl));
	aml_dma_linux_par_destroy(&dma);

	/* move data around */
	assert(!aml_dma_linux_par_create(&dma, 1));
	struct aml_dma_request *requests[16];
	for (int i = 0; i < 16; i++) {
		size_t sz = isz/16;
		size_t off = i*sz;
		void *dptr = (void *)&(idest[off]);
		void *sptr = (void *)&(isrc[off]);

		assert(!aml_dma_async_copy(dma, &requests[i],
					   AML_DMA_REQUEST_TYPE_PTR,
					   dptr, sptr, sz*sizeof(int)));
		assert(requests[i] != NULL);
	}
	for(int i = 0; i < 16; i++)
		assert(!aml_dma_wait(dma, &requests[i]));

	assert(!memcmp(isrc, idest, isz*sizeof(int)));

	/* delete everything */
	aml_dma_linux_par_destroy(&dma);
	aml_layout_dense_destroy(&idl);
	aml_layout_dense_destroy(&isl);
	aml_finalize();
	return 0;
}
