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

#define TILESIZE (2)
#define NBTILES (4)

int main(int argc, char *argv[])
{
 	AML_TILING_1D_DECL(tiling);
	AML_DMA_LINUX_PAR_DECL(dma);
	AML_SCRATCH_SEQ_DECL(scratch);
	void *dst, *src;

	/* library initialization */
	aml_init(&argc, &argv);

	/* initialize all the supporting struct */
	assert(!aml_tiling_init(&tiling, AML_TILING_TYPE_1D, TILESIZE*PAGE_SIZE,
				TILESIZE*PAGE_SIZE*NBTILES));

	size_t maxrequests = NBTILES;
	size_t maxthreads = 4;
	assert(!aml_dma_linux_par_init(&dma, maxrequests, maxthreads));

	/* allocate some memory */
	assert(aml_area_malloc(aml_area_host_private, &src, TILESIZE*PAGE_SIZE*NBTILES, 0) == AML_AREA_SUCCESS);
	assert(src != NULL);

	memset(src, 42, TILESIZE*PAGE_SIZE*NBTILES);

	/* create scratchpad */
	assert(!aml_scratch_seq_init(&scratch, aml_area_host_private, aml_area_host_private, &dma, &tiling,
				     (size_t)NBTILES, (size_t)NBTILES));
	dst = aml_scratch_baseptr(&scratch);
	/* move some stuff */
	for(int i = 0; i < NBTILES; i++)
	{
		int di, si;
		void *dp, *sp;
		aml_scratch_pull(&scratch, dst, &di, src, i);
	
		dp = aml_tiling_tilestart(&tiling, dst, di);
		sp = aml_tiling_tilestart(&tiling, src, i);

		assert(!memcmp(sp, dp, TILESIZE*PAGE_SIZE));

		memset(dp, 33, TILESIZE*PAGE_SIZE);
	
		aml_scratch_push(&scratch, src, &si, dst, di);
		assert(si == i);

		sp = aml_tiling_tilestart(&tiling, src, si);

		assert(!memcmp(sp, dp, TILESIZE*PAGE_SIZE));
	}

	/* delete everything */
	aml_scratch_seq_destroy(&scratch);
	aml_dma_linux_par_destroy(&dma);
	aml_area_free(aml_area_host_private, src);
	aml_tiling_destroy(&tiling, AML_TILING_TYPE_1D);

	aml_finalize();
	return 0;
}
