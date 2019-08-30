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
#include "aml/scratch/seq.h"
#include "aml/tiling/1d.h"
#include <assert.h>

#define TILESIZE (2)
#define NBTILES (4)

int main(int argc, char *argv[])
{
	struct aml_tiling *tiling;
	struct aml_dma *dma;
	struct aml_scratch *scratch;
	void *dst, *src;

	/* library initialization */
	aml_init(&argc, &argv);

	/* initialize all the supporting struct */
	assert(!aml_tiling_1d_create(&tiling, TILESIZE*_SC_PAGE_SIZE,
				     TILESIZE*_SC_PAGE_SIZE*NBTILES));

	size_t maxrequests = NBTILES;
	assert(!aml_dma_linux_par_create(&dma, maxrequests));

	/* allocate some memory */
	src = aml_area_mmap(&aml_area_linux, TILESIZE*_SC_PAGE_SIZE*NBTILES, NULL);
	assert(src != NULL);

	memset(src, 42, TILESIZE*_SC_PAGE_SIZE*NBTILES);

	/* create scratchpad */
	assert(!aml_scratch_seq_create(&scratch, &aml_area_linux, &aml_area_linux, dma, tiling,
				     (size_t)NBTILES, (size_t)NBTILES));
	dst = aml_scratch_baseptr(scratch);
	/* move some stuff */
	for(int i = 0; i < NBTILES; i++)
	{
		int di, si;
		void *dp, *sp;
		aml_scratch_pull(scratch, dst, &di, src, i);
	
		dp = aml_tiling_tilestart(tiling, dst, di);
		sp = aml_tiling_tilestart(tiling, src, i);

		assert(!memcmp(sp, dp, TILESIZE*_SC_PAGE_SIZE));

		memset(dp, 33, TILESIZE*_SC_PAGE_SIZE);
	
		aml_scratch_push(scratch, src, &si, dst, di);
		assert(si == i);

		sp = aml_tiling_tilestart(tiling, src, si);

		assert(!memcmp(sp, dp, TILESIZE*_SC_PAGE_SIZE));
	}

	/* delete everything */
	aml_scratch_seq_destroy(&scratch);
	aml_dma_linux_par_destroy(&dma);
	aml_area_munmap(&aml_area_linux, dst, TILESIZE*_SC_PAGE_SIZE*NBTILES);
	aml_area_munmap(&aml_area_linux, src, TILESIZE*_SC_PAGE_SIZE*NBTILES);
	aml_tiling_1d_destroy(&tiling);

	aml_finalize();
	return 0;
}
