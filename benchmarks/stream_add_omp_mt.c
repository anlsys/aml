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
#include <errno.h>
#include <omp.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#include "utils.h"

#define ITER 10
#define CHUNKING 4

size_t numthreads, tilesz, esz;
unsigned long *a, *b, *c;
AML_TILING_1D_DECL(tiling);
struct aml_area *slow, *fast;
AML_SCRATCH_SEQ_DECL(sa);
AML_SCRATCH_SEQ_DECL(sb);

int kernel(unsigned long *a, unsigned long *b, unsigned long *c, size_t n)
{
	#pragma omp parallel for
	for(size_t i = 0; i < n; i++)
		c[i] = a[i] + b[i];
	return 0;
}

int main(int argc, char *argv[])
{
	AML_DMA_LINUX_PAR_DECL(dma);
	struct aml_bitmap slowb, fastb;
	aml_init(&argc, &argv);
	assert(argc == 4);

	log_init(argv[0]);
	assert(aml_bitmap_from_string(&fastb, argv[1]) == 0);
	assert(aml_bitmap_from_string(&slowb, argv[2]) == 0);
	unsigned long memsize = 1UL << atoi(argv[3]);

	/* use openmp env to figure out how many threads we want
	 * (we actually use 3x as much)
	 */
	#pragma omp parallel
	{
		numthreads = omp_get_num_threads();
		tilesz = memsize/(numthreads*CHUNKING);
		esz = tilesz/sizeof(unsigned long);
	}

	/* initialize all the supporting struct */
	assert(!aml_tiling_init(&tiling, AML_TILING_TYPE_1D, tilesz, memsize));
	aml_area_linux_create(&slow, AML_AREA_LINUX_MMAP_FLAG_PRIVATE,
				     &slowb, AML_AREA_LINUX_BINDING_FLAG_BIND);
	assert(slow != NULL);
	aml_area_linux_create(&fast, AML_AREA_LINUX_MMAP_FLAG_PRIVATE,
				     &fastb, AML_AREA_LINUX_BINDING_FLAG_BIND);
	assert(fast != NULL);
	assert(!aml_dma_linux_par_init(&dma, numthreads*2, numthreads));
	assert(!aml_scratch_seq_init(&sa, &fast, &slow, &dma, &tiling,
				     (size_t)2*numthreads, (size_t)1));
	assert(!aml_scratch_seq_init(&sb, &fast, &slow, &dma, &tiling,
				     (size_t)2*numthreads, (size_t)1));

	/* allocation */
	a = aml_area_mmap(slow, NULL, memsize);
	b = aml_area_mmap(slow, NULL, memsize);
	c = aml_area_mmap(fast, NULL, memsize);
	assert(a != NULL && b != NULL && c != NULL);

	unsigned long esize = memsize/sizeof(unsigned long);
	for(unsigned long i = 0; i < esize; i++) {
		a[i] = i;
		b[i] = esize - i;
		c[i] = 0;
	}

	/* run kernel */
	int i, ai, bi, oldai, oldbi;
	unsigned long *ap, *bp;
	void *abaseptr, *bbaseptr;
	ap = aml_tiling_tilestart(&tiling, a, 0);
	bp = aml_tiling_tilestart(&tiling, b, 0);
	abaseptr = aml_scratch_baseptr(&sa);
	bbaseptr = aml_scratch_baseptr(&sb);
	ai = -1; bi = -1;
	for(i = 0; i < (memsize/tilesz) -1; i++) {
		struct aml_scratch_request *ar, *br;
		oldai = ai; oldbi = bi;
		aml_scratch_async_pull(&sa, &ar, abaseptr, &ai, a, i+1);
		aml_scratch_async_pull(&sb, &br, bbaseptr, &bi, b, i+1);
		kernel(ap, bp, &c[i*esz], esz);
		aml_scratch_wait(&sa, ar);
		aml_scratch_wait(&sb, br);
		ap = aml_tiling_tilestart(&tiling, abaseptr, ai);
		bp = aml_tiling_tilestart(&tiling, bbaseptr, bi);
		aml_scratch_release(&sa, oldai);
		aml_scratch_release(&sb, oldbi);
	}
	kernel(ap, bp, &c[i*esz], esz);

	/* validate */
	for(unsigned long i = 0; i < esize; i++) {
		assert(c[i] == esize);
	}

	aml_scratch_seq_destroy(&sa);
	aml_scratch_seq_destroy(&sb);
	aml_dma_linux_par_fini(&dma);
	aml_area_munmap(slow, a, memsize);
	aml_area_munmap(slow, b, memsize);
	aml_area_munmap(fast, c, memsize);
	aml_area_linux_destroy(&slow);
	aml_area_linux_destroy(&fast);
	aml_tiling_destroy(&tiling, AML_TILING_TYPE_1D);
	aml_finalize();
	return 0;
}
