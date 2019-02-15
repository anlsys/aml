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
#include <aml.h>
#include <stdlib.h>

#define ITER 10
#define MEMSIZE (1UL<<26)
#define PHASES 20
#define CHUNKING 4

int kernel(unsigned long *a, unsigned long *b, unsigned long *c, size_t n)
{
	size_t i;
	for(i = 0; i < n; i++)
		c[i] = a[i] + b[i];
	return 0;
}

int main(int argc, char *argv[])
{
	assert(argc == 1);
	aml_init(&argc, &argv);

	/* we want to back our array on the slow node and use the fast node as
	 * a faster buffer.
	 */
	struct aml_area slow, fast;
	int type = AML_AREA_TYPE_REGULAR;
	assert(!aml_area_from_nodestring(&slow, type, "all"));
	assert(!aml_area_from_nodestring(&fast, type, "all"));

	struct aml_dma dma;
	assert(!aml_dma_init(&dma, 0));

	void *a, *b, *c;

	/* describe the allocation */
	size_t chunk_msz, esz;
	int numthreads;

	#pragma omp parallel
	{
		numthreads = omp_get_num_threads();
		chunk_msz = MEMSIZE/(numthreads*CHUNKING);
		esz = chunk_msz/sizeof(unsigned long);
	}
	a = aml_area_malloc(&slow, MEMSIZE);
	b = aml_area_malloc(&slow, MEMSIZE);
	c = aml_area_malloc(&slow, MEMSIZE);
	assert(a != NULL && b != NULL && c != NULL);

	/* create virtually accessible address range, backed by slow memory */
	unsigned long *wa = (unsigned long*)a;
	unsigned long *wb = (unsigned long*)b;
	unsigned long *wc = (unsigned long*)c;
	unsigned long esize = MEMSIZE/sizeof(unsigned long);
	for(unsigned long i = 0; i < esize; i++) {
		wa[i] = i;
		wb[i] = esize - i;
		wc[i] = 0;
	}

	/* run kernel */
	#pragma omp parallel
	#pragma omp single nowait
	{
		for(unsigned long i = 0; i < numthreads*CHUNKING; i++) {
			#pragma omp task depend(inout: wa[i*esz:esz])
			assert(!aml_dma_move(&dma, &fast, &slow, &wa[i*esz], esz));
			#pragma omp task depend(inout: wb[i*esz:esz])
			assert(!aml_dma_move(&dma, &fast, &slow, &wb[i*esz], esz));
			#pragma omp task depend(inout: wc[i*esz:esz])
			assert(!aml_dma_move(&dma, &fast, &slow, &wc[i*esz], esz));
			#pragma omp task depend(in: wa[i*esz:esz], wb[i*esz:esz]) depend(out: wc[i*esz:esz])
			kernel(&wa[i*esz], &wb[i*esz], &wc[i*esz], esz);
			#pragma omp task depend(inout: wa[i*esz:esz])
			assert(!aml_dma_move(&dma, &slow, &fast, &wa[i*esz], esz));
			#pragma omp task depend(inout: wb[i*esz:esz])
			assert(!aml_dma_move(&dma, &slow, &fast, &wb[i*esz], esz));
			#pragma omp task depend(inout: wc[i*esz:esz])
			assert(!aml_dma_move(&dma, &slow, &fast, &wc[i*esz], esz));
		}
	}

	/* validate */
	for(unsigned long i = 0; i < esize; i++) {
		assert(wc[i] == esize);
	}

	aml_area_free(&slow, a);
	aml_area_free(&slow, b);
	aml_area_free(&slow, c);
	aml_area_destroy(&slow);
	aml_area_destroy(&fast);
	aml_dma_destroy(&dma);
	aml_finalize();
	return 0;
}
