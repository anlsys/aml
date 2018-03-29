#include <aml.h>
#include <assert.h>
#include <errno.h>
#include <omp.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define ITER 10
#define MEMSIZE (1UL<<20)
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
	printf("%p = %p + %p [%zi]\n",c,a,b,n);
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
	unsigned long *ap, *bp;
	void *abaseptr, *bbaseptr;
	struct winfo *wi = arg;
	offset = wi->tid*CHUNKING;
	ap = aml_tiling_tilestart(&tiling, a, offset);
	bp = aml_tiling_tilestart(&tiling, b, offset);
	abaseptr = aml_scratch_baseptr(&sa);
	bbaseptr = aml_scratch_baseptr(&sb);
	ai = -1; bi = -1;
	for(i = 0; i < CHUNKING-1; i++) {
		struct aml_scratch_request *ar, *br;
		oldai = ai; oldbi = bi;
		aml_scratch_async_pull(&sa, &ar, abaseptr, &ai, a, offset+i+1);
		aml_scratch_async_pull(&sb, &br, bbaseptr, &bi, b, offset+i+1);
		kernel(ap, bp, &c[(offset+i)*esz], esz);
		aml_scratch_wait(&sa, ar);
		aml_scratch_wait(&sb, br);
		ap = aml_tiling_tilestart(&tiling, abaseptr, ai);
		bp = aml_tiling_tilestart(&tiling, bbaseptr, bi);
		aml_scratch_release(&sa, oldai);
		aml_scratch_release(&sb, oldbi);
	}
	kernel(ap, bp, &c[(offset+i)*esz], esz);

	return arg;
}
int main(int argc, char *argv[])
{
	AML_BINDING_SINGLE_DECL(binding);
	AML_ARENA_JEMALLOC_DECL(arena);
	AML_DMA_LINUX_SEQ_DECL(dma);
	unsigned long nodemask[AML_NODEMASK_SZ];
	aml_init(&argc, &argv);
	assert(argc == 1);

	/* use openmp env to figure out how many threads we want
	 * (we actually use 3x as much)
	 */
	#pragma omp parallel
	{
		numthreads = omp_get_num_threads();
		tilesz = MEMSIZE/(numthreads*CHUNKING);
		esz = tilesz/sizeof(unsigned long);
	}

	/* initialize all the supporting struct */
	assert(!aml_binding_init(&binding, AML_BINDING_TYPE_SINGLE, 0));
	assert(!aml_tiling_init(&tiling, AML_TILING_TYPE_1D, tilesz, MEMSIZE));
	AML_NODEMASK_ZERO(nodemask);
	AML_NODEMASK_SET(nodemask, 0);
	assert(!aml_arena_jemalloc_init(&arena, AML_ARENA_JEMALLOC_TYPE_REGULAR));

	assert(!aml_area_linux_init(&slow,
				    AML_AREA_LINUX_MANAGER_TYPE_SINGLE,
				    AML_AREA_LINUX_MBIND_TYPE_REGULAR,
				    AML_AREA_LINUX_MMAP_TYPE_ANONYMOUS,
				    &arena, MPOL_BIND, nodemask));
	assert(!aml_area_linux_init(&fast,
				    AML_AREA_LINUX_MANAGER_TYPE_SINGLE,
				    AML_AREA_LINUX_MBIND_TYPE_REGULAR,
				    AML_AREA_LINUX_MMAP_TYPE_ANONYMOUS,
				    &arena, MPOL_BIND, nodemask));
	assert(!aml_dma_linux_seq_init(&dma, numthreads*2));
	assert(!aml_scratch_par_init(&sa, &fast, &slow, &dma, &tiling,
				     2*numthreads, numthreads));
	assert(!aml_scratch_par_init(&sb, &fast, &slow, &dma, &tiling,
				     2*numthreads, numthreads));

	/* allocation */
	a = aml_area_malloc(&slow, MEMSIZE);
	b = aml_area_malloc(&slow, MEMSIZE);
	c = aml_area_malloc(&fast, MEMSIZE);
	assert(a != NULL && b != NULL && c != NULL);

	unsigned long esize = MEMSIZE/sizeof(unsigned long);
	for(unsigned long i = 0; i < esize; i++) {
		a[i] = i;
		b[i] = esize - i;
		c[i] = 0;
	}

	/* run kernel */
	struct winfo *wis = aml_area_calloc(&slow, numthreads, sizeof(struct winfo));
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
	aml_area_free(&slow, a);
	aml_area_free(&slow, b);
	aml_area_free(&slow, c);
	aml_area_linux_destroy(&slow);
	aml_area_linux_destroy(&fast);
	aml_tiling_destroy(&tiling, AML_TILING_TYPE_1D);
	aml_binding_destroy(&binding, AML_BINDING_TYPE_SINGLE);
	aml_finalize();
	return 0;
}
