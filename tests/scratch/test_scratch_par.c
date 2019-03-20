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
#include "aml/dma/linux-seq.h"
#include "aml/scratch/par.h"
#include <assert.h>

#define TILESIZE (2)
#define NBTILES (4)

int main(int argc, char *argv[])
{
	AML_BINDING_SINGLE_DECL(binding);
	AML_TILING_1D_DECL(tiling);
	AML_ARENA_JEMALLOC_DECL(arena);
	AML_AREA_LINUX_DECL(area);
	AML_DMA_LINUX_SEQ_DECL(dma);
	AML_SCRATCH_PAR_DECL(scratch);
	struct aml_bitmap nodemask;
	void *dst, *src;

	/* library initialization */
	aml_init(&argc, &argv);

	/* initialize all the supporting struct */
	assert(!aml_binding_init(&binding, AML_BINDING_TYPE_SINGLE, 0));
	assert(!aml_tiling_init(&tiling, AML_TILING_TYPE_1D, TILESIZE*PAGE_SIZE,
				TILESIZE*PAGE_SIZE*NBTILES));
	aml_bitmap_zero(&nodemask);
	aml_bitmap_set(&nodemask, 0);
	assert(!aml_arena_jemalloc_init(&arena, AML_ARENA_JEMALLOC_TYPE_REGULAR));

	assert(!aml_area_linux_init(&area,
				    AML_AREA_LINUX_MANAGER_TYPE_SINGLE,
				    AML_AREA_LINUX_MBIND_TYPE_REGULAR,
				    AML_AREA_LINUX_MMAP_TYPE_ANONYMOUS,
				    &arena, MPOL_BIND, &nodemask));

	size_t maxrequests = NBTILES;
	assert(!aml_dma_linux_seq_init(&dma, maxrequests));

	/* allocate some memory */
	src = aml_area_malloc(&area, TILESIZE*PAGE_SIZE*NBTILES);
	assert(src != NULL);

	memset(src, 42, TILESIZE*PAGE_SIZE*NBTILES);

	/* create scratchpad */
	assert(!aml_scratch_par_init(&scratch, &area, &area, &dma, &tiling,
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
	aml_scratch_par_destroy(&scratch);
	aml_dma_linux_seq_destroy(&dma);
	aml_area_free(&area, dst);
	aml_area_free(&area, src);
	aml_area_linux_destroy(&area);
	aml_tiling_destroy(&tiling, AML_TILING_TYPE_1D);
	aml_binding_destroy(&binding, AML_BINDING_TYPE_SINGLE);

	aml_finalize();
	return 0;
}
