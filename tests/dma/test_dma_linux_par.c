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
#include "aml/dma/linux-par.h"
#include "aml/tiling/1d.h"
#include <assert.h>

#define TILESIZE (2)
#define NBTILES (4)

int main(int argc, char *argv[])
{
	struct aml_tiling *tiling;
	struct aml_dma *dma;
	void *dst, *src;

	/* library initialization */
	aml_init(&argc, &argv);

	/* initialize all the supporting struct */
	assert(!aml_tiling_1d_create(&tiling, TILESIZE*_SC_PAGE_SIZE,
				     TILESIZE*_SC_PAGE_SIZE*NBTILES));
	size_t maxrequests = NBTILES;
	size_t maxthreads = 4;
	assert(!aml_dma_linux_par_create(&dma, maxrequests, maxthreads));

	/* allocate some memory */
	src = aml_area_mmap(&aml_area_linux, NULL, TILESIZE*_SC_PAGE_SIZE*NBTILES);
	assert(src != NULL);
	dst = aml_area_mmap(&aml_area_linux, NULL, TILESIZE*_SC_PAGE_SIZE*NBTILES);
	assert(dst != NULL);

	memset(src, 42, TILESIZE*_SC_PAGE_SIZE*NBTILES);
	memset(dst, 24, TILESIZE*_SC_PAGE_SIZE*NBTILES);

	/* move some stuff by copy */
	for(int i = 0; i < NBTILES; i++)
		aml_dma_copy(dma, tiling, dst, i, tiling, src, i);

	assert(!memcmp(src, dst, TILESIZE*_SC_PAGE_SIZE*NBTILES));

	/* delete everything */
	aml_dma_linux_par_destroy(&dma);
	aml_area_munmap(&aml_area_linux, dst, TILESIZE*_SC_PAGE_SIZE*NBTILES);
	aml_area_munmap(&aml_area_linux, src, TILESIZE*_SC_PAGE_SIZE*NBTILES);
	aml_tiling_1d_destroy(&tiling);

	aml_finalize();
	return 0;
}
