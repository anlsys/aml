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
#include "aml/area/linux.h"
#include "aml/dma/linux-seq.h"
#include "aml/scratch/par.h"
#include "aml/tiling/1d.h"
#include "utils.h"

#define ITER 10
#define CHUNKING 4

size_t numthreads, tilesz, esz;
unsigned long *a, *b, *c;
struct aml_area *slow, *fast;
struct aml_tiling *tiling;
struct aml_scratch *sa;
struct aml_scratch *sb;

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
	ap = aml_tiling_tilestart(tiling, a, offset);
	bp = aml_tiling_tilestart(tiling, b, offset);
	cp = aml_tiling_tilestart(tiling, c, offset);
	abaseptr = aml_scratch_baseptr(sa);
	bbaseptr = aml_scratch_baseptr(sb);
	ai = -1; bi = -1;
	for(i = 0; i < CHUNKING-1; i++) {
		struct aml_scratch_request *ar, *br;
		oldai = ai; oldbi = bi;
		aml_scratch_async_pull(sa, &ar, abaseptr, &ai, a, offset+i+1);
		aml_scratch_async_pull(sb, &br, bbaseptr, &bi, b, offset+i+1);
		kernel(ap, bp, cp, esz);
		aml_scratch_wait(sa, ar);
		aml_scratch_wait(sb, br);
		ap = aml_tiling_tilestart(tiling, abaseptr, ai);
		bp = aml_tiling_tilestart(tiling, bbaseptr, bi);
		cp = aml_tiling_tilestart(tiling, c, offset+i+1);
		aml_scratch_release(sa, oldai);
		aml_scratch_release(sb, oldbi);
	}
	kernel(ap, bp, cp, esz);

	return arg;
}
int main(int argc, char *argv[])
{
	struct aml_dma *dma;
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
	assert(!aml_tiling_1d_create(&tiling, tilesz, memsize));
	aml_area_linux_create(&slow, &slowb, AML_AREA_LINUX_POLICY_BIND);
	assert(slow != NULL);
	aml_area_linux_create(&fast, &fastb, AML_AREA_LINUX_POLICY_BIND);
	assert(fast != NULL);
	assert(!aml_dma_linux_seq_create(dma, (size_t)numthreads*4, NULL));
	assert(!aml_scratch_par_create(&sa, fast, slow, dma, tiling,
				     (size_t)2*numthreads, (size_t)numthreads));
	assert(!aml_scratch_par_create(&sb, fast, slow, dma, tiling,
				     (size_t)2*numthreads, (size_t)numthreads));

	/* allocation */
	a = aml_area_mmap(slow, memsize, NULL);
	b = aml_area_mmap(slow, memsize, NULL);
	c = aml_area_mmap(fast, memsize, NULL);
	assert(a != NULL && b != NULL && c != NULL);

	unsigned long esize = memsize/sizeof(unsigned long);
	for(unsigned long i = 0; i < esize; i++) {
		a[i] = i;
		b[i] = esize - i;
		c[i] = 0;
	}

	/* run kernel */
	struct winfo *wis = aml_area_mmap(slow, numthreads * sizeof(struct winfo), NULL);
	for(unsigned long i = 0; i < numthreads; i++) {
		wis[i].tid = i;
		pthread_create(&wis[i].th, NULL, &th_work, (void*)&wis[i]);
	}
	for(unsigned long j = 0; j < numthreads; j++) {
		pthread_join(wis[j].th, NULL);
	}
	aml_area_munmap(slow, wis, numthreads * sizeof(struct winfo));

	/* validate */
	for(unsigned long i = 0; i < esize; i++) {
		assert(c[i] == esize);
	}

	aml_scratch_par_destroy(&sa);
	aml_scratch_par_destroy(&sb);
	aml_dma_linux_seq_destroy(&dma);
	aml_area_munmap(slow, a, memsize);
	aml_area_munmap(slow, b, memsize);
	aml_area_munmap(fast, c, memsize);
	aml_area_linux_destroy(&slow);
	aml_area_linux_destroy(&fast);
	aml_tiling_1d_destroy(&tiling);
	aml_finalize();
	return 0;
}
