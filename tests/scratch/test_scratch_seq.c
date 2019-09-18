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
#include "aml/layout/dense.h"
#include "aml/tiling/resize.h"
#include "aml/dma/linux-par.h"
#include "aml/scratch/seq.h"
#include <assert.h>

#define TILESIZE (2)
#define NBTILES (4)

int main(int argc, char *argv[])
{
	struct aml_layout *source;
	struct aml_layout *scratch_layout;
	struct aml_tiling *src_tiling;
	struct aml_tiling *scratch_tiling;
	struct aml_dma *dma;
	struct aml_scratch *scratch;
	void *dst, *src;

	/* library initialization */
	aml_init(&argc, &argv);

	/* initialize all the supporting struct */
	size_t size = TILESIZE*_SC_PAGE_SIZE*NBTILES;
	size_t tsize = TILESIZE*_SC_PAGE_SIZE;

	src = aml_area_mmap(&aml_area_linux, size, NULL);
	dst = aml_area_mmap(&aml_area_linux, size, NULL);
	assert(src != NULL && dst != NULL);

	assert(!aml_layout_dense_create(&source, src,
					AML_LAYOUT_ORDER_COLUMN_MAJOR,
					sizeof(char), 1, &size, NULL, NULL));
	assert(!aml_layout_dense_create(&scratch_layout, dst,
					AML_LAYOUT_ORDER_COLUMN_MAJOR,
					sizeof(char), 1, &size, NULL, NULL));
	assert(!aml_tiling_resize_create(&src_tiling,
					 AML_TILING_ORDER_COLUMN_MAJOR,
					 source, 1, &tsize));
	assert(!aml_tiling_resize_create(&scratch_tiling,
					 AML_TILING_ORDER_COLUMN_MAJOR,
					 scratch_layout, 1, &tsize));
	size_t maxrequests = NBTILES;

	assert(!aml_dma_linux_par_create(&dma, maxrequests, NULL, NULL));

	/* setup some initial values in the memory */
	for (size_t i = 0; i < NBTILES; i++) {
		char *s, *d;

		s = &((char *)src)[i * tsize];
		d = &((char *)dst)[i * tsize];
		memset((void *)s, (char)i, TILESIZE*_SC_PAGE_SIZE);
		memset((void *)d, (char)NBTILES + i, TILESIZE*_SC_PAGE_SIZE);
	}

	/* create scratchpad */
	assert(!aml_scratch_seq_create(&scratch, dma, src_tiling,
				       scratch_tiling, maxrequests));
	/* move some stuff */
	for (size_t i = 0; i < NBTILES; i++) {
		int di, si;
		void *dp, *sp;
		struct aml_layout *sl, *dl;

		si = aml_tiling_tileid(src_tiling, &i);
		sl = aml_tiling_index_byid(src_tiling, si);
		assert(sl != NULL);

		assert(!aml_scratch_pull(scratch, &dl, &di, sl, si));
		assert(dl != NULL && di != -1);

		sp = aml_layout_deref(sl, (size_t[]){0});
		dp = aml_layout_deref(dl, (size_t[]){0});

		assert(!memcmp(sp, dp, TILESIZE*_SC_PAGE_SIZE));
		assert(*(char *)sp == (char)i);

		memset(dp, NBTILES+i, TILESIZE*_SC_PAGE_SIZE);

		free(sl);
		assert(!aml_scratch_push(scratch, &sl, &si, dl, di));
		assert(sl != NULL);
		assert(si == (int)i);

		free(dl);
		sp = aml_layout_deref(sl, (size_t[]){0});
		assert(!memcmp(sp, dp, TILESIZE*_SC_PAGE_SIZE));
		free(sl);
	}

	/* delete everything */
	aml_scratch_seq_destroy(&scratch);
	aml_dma_linux_par_destroy(&dma);
	aml_tiling_resize_destroy(&src_tiling);
	aml_tiling_resize_destroy(&scratch_tiling);
	aml_layout_dense_destroy(&source);
	aml_layout_dense_destroy(&scratch_layout);
	aml_area_munmap(&aml_area_linux, dst, TILESIZE*_SC_PAGE_SIZE*NBTILES);
	aml_area_munmap(&aml_area_linux, src, TILESIZE*_SC_PAGE_SIZE*NBTILES);
	aml_finalize();
	return 0;
}
