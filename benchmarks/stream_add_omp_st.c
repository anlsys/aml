#include <aml.h>
#include <assert.h>
#include <errno.h>
#include <omp.h>
#include <pthread.h>
#include <stdlib.h>

#include "utils.h"

#define ITER 10
#define CHUNKING 4

size_t numthreads, tilesz, esz;
unsigned long *a, *b, *c;
AML_TILING_1D_DECL(tiling);
AML_AREA_LINUX_DECL(slow);
AML_AREA_LINUX_DECL(fast);
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

void do_work(unsigned long tid)
{

	int offset, i, ai, bi, oldai, oldbi;
	unsigned long *ap, *bp, *cp;
	void *abaseptr, *bbaseptr;
	offset = tid*CHUNKING;
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
}

int main(int argc, char *argv[])
{
	AML_ARENA_JEMALLOC_DECL(arns);
	AML_ARENA_JEMALLOC_DECL(arnf);
	AML_DMA_LINUX_SEQ_DECL(dma);
	unsigned long nodemask[AML_NODEMASK_SZ];
	struct bitmask *slowb, *fastb;
	aml_init(&argc, &argv);
	assert(argc == 4);

	log_init(argv[0]);
	fastb = numa_parse_nodestring_all(argv[1]);
	slowb = numa_parse_nodestring_all(argv[2]);
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

	assert(!aml_arena_jemalloc_init(&arns, AML_ARENA_JEMALLOC_TYPE_REGULAR));
	assert(!aml_area_linux_init(&slow,
				    AML_AREA_LINUX_MANAGER_TYPE_SINGLE,
				    AML_AREA_LINUX_MBIND_TYPE_REGULAR,
				    AML_AREA_LINUX_MMAP_TYPE_ANONYMOUS,
				    &arns, MPOL_BIND, slowb->maskp));
	assert(!aml_arena_jemalloc_init(&arnf, AML_ARENA_JEMALLOC_TYPE_REGULAR));
	assert(!aml_area_linux_init(&fast,
				    AML_AREA_LINUX_MANAGER_TYPE_SINGLE,
				    AML_AREA_LINUX_MBIND_TYPE_REGULAR,
				    AML_AREA_LINUX_MMAP_TYPE_ANONYMOUS,
				    &arnf, MPOL_BIND, fastb->maskp));
	assert(!aml_dma_linux_seq_init(&dma, numthreads*2));
	assert(!aml_scratch_par_init(&sa, &fast, &slow, &dma, &tiling,
				     2*numthreads, numthreads));
	assert(!aml_scratch_par_init(&sb, &fast, &slow, &dma, &tiling,
				     2*numthreads, numthreads));

	/* allocation */
	a = aml_area_malloc(&slow, memsize);
	b = aml_area_malloc(&slow, memsize);
	c = aml_area_malloc(&fast, memsize);
	assert(a != NULL && b != NULL && c != NULL);

	unsigned long esize = memsize/sizeof(unsigned long);
	for(unsigned long i = 0; i < esize; i++) {
		a[i] = i;
		b[i] = esize - i;
		c[i] = 0;
	}

	/* run kernel */
	#pragma omp parallel for
	for(unsigned long i = 0; i < numthreads; i++) {
		do_work(i);
	}

	/* validate */
	for(unsigned long i = 0; i < esize; i++) {
		assert(c[i] == esize);
	}

	aml_scratch_par_destroy(&sa);
	aml_scratch_par_destroy(&sb);
	aml_dma_linux_seq_destroy(&dma);
	aml_area_free(&slow, a);
	aml_area_free(&slow, b);
	aml_area_free(&fast, c);
	aml_area_linux_destroy(&slow);
	aml_area_linux_destroy(&fast);
	aml_tiling_destroy(&tiling, AML_TILING_TYPE_1D);
	aml_finalize();
	return 0;
}