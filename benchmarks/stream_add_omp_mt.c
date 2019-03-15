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
	struct aml_bitmap slow_b, fast_b;
	aml_bitmap_zero(&slow_b);
	aml_bitmap_zero(&fast_b);
	aml_bitmap_set(&slow_b, 0);
	aml_bitmap_set(&fast_b, 1);
	struct aml_area * slow = aml_local_area_create(aml_area_host_private, &slow_b, 0);
	struct aml_area * fast = aml_local_area_create(aml_area_host_private, &fast_b, 0);

	AML_DMA_LINUX_PAR_DECL(dma);
	aml_init(&argc, &argv);
	assert(argc == 4);

	log_init(argv[0]);	
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

	assert(!aml_dma_linux_par_init(&dma, numthreads*2, numthreads));
	assert(!aml_scratch_seq_init(&sa, &fast, &slow, &dma, &tiling,
				     (size_t)2*numthreads, (size_t)1));
	assert(!aml_scratch_seq_init(&sb, &fast, &slow, &dma, &tiling,
				     (size_t)2*numthreads, (size_t)1));

	/* allocation */
	assert(aml_area_malloc(slow, (void**)(&a), memsize, 0) == AML_AREA_SUCCESS);
	assert(aml_area_malloc(slow, (void**)(&b), memsize, 0) == AML_AREA_SUCCESS);
	assert(aml_area_malloc(fast, (void**)(&c), memsize, 0) == AML_AREA_SUCCESS);
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
	aml_dma_linux_par_destroy(&dma);
	aml_area_free(slow, a);
	aml_area_free(slow, b);
	aml_area_free(fast, c);
	aml_local_area_destroy(slow);
	aml_local_area_destroy(fast);
	aml_tiling_destroy(&tiling, AML_TILING_TYPE_1D);
	aml_finalize();
	return 0;
}
