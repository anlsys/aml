/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
*******************************************************************************/

#include <assert.h>
#include <errno.h>
#include <omp.h>
#include <pthread.h>
#include <stdlib.h>

#include "aml.h"
#include "utils.h"

#define ITER 10
#define CHUNKING 4

size_t numthreads, tilesz, esz;
unsigned long *a, *b, *c;
AML_TILING_1D_DECL(tiling);
AML_SCRATCH_PAR_DECL(sa);
AML_SCRATCH_PAR_DECL(sb);

int kernel(unsigned long *a, unsigned long *b, unsigned long *c, size_t n)
{
	size_t i;
	debug("%p = %p + %p [%zi]\n",c,a,b,n);
	for(i = 0; i < n; i++)
		c[i] = a[i] + b[i];
	return 0;
}

struct winfo {
	int tid;
	pthread_t th;
};

void *th_work(void *arg)
{

	int offset, i, ai, bi, oldai, oldbi;
	unsigned long *ap, *bp, *cp;
	void *abaseptr, *bbaseptr;
	struct winfo *wi = arg;
	offset = wi->tid*CHUNKING;
	ap = aml_tiling_tilestart(&tiling, a, offset);
	bp = aml_tiling_tilestart(&tiling, b, offset);
	cp = aml_tiling_tilestart(&tiling, c, offset);
	abaseptr = aml_scratch_baseptr(&sa);
	bbaseptr = aml_scratch_baseptr(&sb);
	ai = -1; bi = -1;
	for(i = 0; i < CHUNKING-1; i++) {
		struct aml_scratch_request *ar, *br;
		oldai = ai; oldbi = bi;
		aml_scratch_async_pull(&sa, &ar, abaseptr, &ai, a, offset+i+1);
		aml_scratch_async_pull(&sb, &br, bbaseptr, &bi, b, offset+i+1);
		kernel(ap, bp, cp, esz);
		aml_scratch_wait(&sa, ar);
		aml_scratch_wait(&sb, br);
		ap = aml_tiling_tilestart(&tiling, abaseptr, ai);
		bp = aml_tiling_tilestart(&tiling, bbaseptr, bi);
		cp = aml_tiling_tilestart(&tiling, c, offset+i+1);
		aml_scratch_release(&sa, oldai);
		aml_scratch_release(&sb, oldbi);
	}
	kernel(ap, bp, cp, esz);

	return arg;
}
int main(int argc, char *argv[])
{
	AML_DMA_LINUX_SEQ_DECL(dma);
	aml_init(&argc, &argv);
	assert(argc == 4);

	log_init(argv[0]);
	unsigned long memsize = 1UL << atoi(argv[3]);

	struct aml_bitmap slow_b, fast_b;
	aml_bitmap_zero(&slow_b);
	aml_bitmap_zero(&fast_b);
	aml_bitmap_set(&slow_b, 0);
	aml_bitmap_set(&fast_b, 1);
	struct aml_area * slow = aml_local_area_create(aml_area_host_private, &slow_b, 0);
	struct aml_area * fast = aml_local_area_create(aml_area_host_private, &fast_b, 0);

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
	assert(!aml_dma_linux_seq_init(&dma, (size_t)numthreads*4));
	assert(!aml_scratch_par_init(&sa, &fast, &slow, &dma, &tiling,
				     (size_t)2*numthreads, (size_t)numthreads));
	assert(!aml_scratch_par_init(&sb, &fast, &slow, &dma, &tiling,
				     (size_t)2*numthreads, (size_t)numthreads));

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
	struct winfo *wis;
	assert(aml_area_malloc(slow, (void**)(&wis), numthreads * sizeof(struct winfo)) == AML_AREA_SUCCESS);
	for(unsigned long i = 0; i < numthreads; i++) {
		wis[i].tid = i;
		pthread_create(&wis[i].th, NULL, &th_work, (void*)&wis[i]);
	}
	for(unsigned long j = 0; j < numthreads; j++) {
		pthread_join(wis[j].th, NULL);
	}
	aml_area_free(&slow, wis);

	/* validate */
	for(unsigned long i = 0; i < esize; i++) {
		assert(c[i] == esize);
	}

	aml_scratch_par_destroy(&sa);
	aml_scratch_par_destroy(&sb);
	aml_dma_linux_seq_destroy(&dma);
	aml_area_free(slow, a);
	aml_area_free(slow, b);
	aml_area_free(fast, c);
	aml_local_area_destroy(slow);
	aml_local_area_destroy(fast);
	aml_tiling_destroy(&tiling, AML_TILING_TYPE_1D);
	aml_finalize();
	return 0;
}
