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
#include "aml/area/linux.h"
#include "aml/dma/linux-seq.h"
#include <assert.h>

#define TILESIZE (2)
#define NBTILES (4)

int main(int argc, char *argv[])
{
	AML_TILING_1D_DECL(tiling);
	AML_DMA_LINUX_SEQ_DECL(dma);
	struct aml_bitmap nodemask;
	void *dst, *src;

	/* library initialization */
	aml_init(&argc, &argv);

	/* initialize all the supporting struct */
	assert(!aml_tiling_init(&tiling, AML_TILING_TYPE_1D, TILESIZE*PAGE_SIZE,
				TILESIZE*PAGE_SIZE*NBTILES));
	aml_bitmap_zero(&nodemask);
	aml_bitmap_set(&nodemask, 0);

	size_t maxrequests = NBTILES;
	assert(!aml_dma_linux_seq_init(&dma, maxrequests));

	/* allocate some memory */
	src = aml_area_mmap(&aml_area_linux, NULL, TILESIZE*PAGE_SIZE*NBTILES);
	assert(src != NULL);
	dst = aml_area_mmap(&aml_area_linux, NULL, TILESIZE*PAGE_SIZE*NBTILES);
	assert(dst != NULL);

	memset(src, 42, TILESIZE*PAGE_SIZE*NBTILES);
	memset(dst, 24, TILESIZE*PAGE_SIZE*NBTILES);

	/* move some stuff by copy */
	for(int i = 0; i < NBTILES; i++)
		aml_dma_copy(&dma, &tiling, dst, i, &tiling, src, i);

	assert(!memcmp(src, dst, TILESIZE*PAGE_SIZE*NBTILES));

	/* delete everything */
	aml_dma_linux_seq_destroy(&dma);
	aml_area_munmap(&aml_area_linux, dst, TILESIZE*PAGE_SIZE*NBTILES);
	aml_area_munmap(&aml_area_linux, src, TILESIZE*PAGE_SIZE*NBTILES);
	aml_tiling_destroy(&tiling, AML_TILING_TYPE_1D);

	aml_finalize();
	return 0;
}
