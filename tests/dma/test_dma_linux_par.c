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

#define PAGE_SIZE 4096
#define TILESIZE (2)
#define NBTILES (4)

int main(int argc, char *argv[])
{
	AML_TILING_1D_DECL(tiling);
	AML_DMA_LINUX_PAR_DECL(dma);
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
	assert(aml_area_malloc(aml_area_host_private, &dst, TILESIZE*PAGE_SIZE*NBTILES, 0) == AML_AREA_SUCCESS);
	assert(dst != NULL);

	memset(src, 42, TILESIZE*PAGE_SIZE*NBTILES);
	memset(dst, 24, TILESIZE*PAGE_SIZE*NBTILES);

	/* move some stuff by copy */
	for(int i = 0; i < NBTILES; i++)
		aml_dma_copy(&dma, &tiling, dst, i, &tiling, src, i);

	assert(!memcmp(src, dst, TILESIZE*PAGE_SIZE*NBTILES));

	/* delete everything */
	aml_dma_linux_par_destroy(&dma);
	aml_area_free(aml_area_host_private, dst);
	aml_area_free(aml_area_host_private, src);
	aml_tiling_destroy(&tiling, AML_TILING_TYPE_1D);

	aml_finalize();
	return 0;
}
