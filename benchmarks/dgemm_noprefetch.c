#include <aml.h>
#include <assert.h>
#include <errno.h>
#include <mkl.h>
#include <omp.h>
#include <pthread.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>

AML_TILING_2D_CONTIG_DECL(tiling);
AML_AREA_LINUX_DECL(slow);
AML_AREA_LINUX_DECL(fast);

size_t memsize, tilesize, N, T;
double *a, *b, *c;
struct timespec start, stop;

void do_work()
{
	int lda = (int)T, ldb, ldc;
	ldb = lda;
	ldc = lda;
	double *ap, *bp, *cp;
	size_t ndims[2];
	aml_tiling_ndims(&tiling, &ndims[0], &ndims[1]);
	size_t aoff, boff, coff;

	for(int j = 0; j < ndims[1]; j++)
	{
		for(int k = 0; k < ndims[1]; k++)
		{
			#pragma omp parallel for
			for(int i = 0; i < ndims[1]; i++)
			{
				aoff = i*ndims[0] + k;
				boff = k*ndims[0] + j;
				coff = i*ndims[0] + j;
				ap = aml_tiling_tilestart(&tiling, ap, aoff);
				bp = aml_tiling_tilestart(&tiling, bp, boff);
				cp = aml_tiling_tilestart(&tiling, cp, coff);
				cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ldc, lda, ldb, 1.0, ap, lda, bp, ldb, 1.0, cp, ldc);
			}
		}
	}
}

int main(int argc, char* argv[])
{
	AML_ARENA_JEMALLOC_DECL(arena);
	AML_DMA_LINUX_SEQ_DECL(dma);
	struct bitmask *slowb, *fastb;
	aml_init(&argc, &argv);
	assert(argc == 5);
	fastb = numa_parse_nodestring_all(argv[1]);
	slowb = numa_parse_nodestring_all(argv[2]);
	N = atol(argv[3]);
	T = atol(argv[4]);
	/* let's not handle messy tile sizes */
	assert(N % T == 0);
	memsize = sizeof(double)*N*N;
	tilesize = sizeof(double)*T*T;

	/* the initial tiling, of 2D square tiles */
	assert(!aml_tiling_init(&tiling, AML_TILING_TYPE_2D_CONTIG,
				tilesize, memsize, N/T , N/T));
	assert(!aml_arena_jemalloc_init(&arena, AML_ARENA_JEMALLOC_TYPE_REGULAR));
	assert(!aml_area_linux_init(&slow,
				    AML_AREA_LINUX_MANAGER_TYPE_SINGLE,
				    AML_AREA_LINUX_MBIND_TYPE_REGULAR,
				    AML_AREA_LINUX_MMAP_TYPE_ANONYMOUS,
				    &arena, MPOL_BIND, slowb->maskp));
	assert(!aml_area_linux_init(&fast,
				    AML_AREA_LINUX_MANAGER_TYPE_SINGLE,
				    AML_AREA_LINUX_MBIND_TYPE_REGULAR,
				    AML_AREA_LINUX_MMAP_TYPE_ANONYMOUS,
				    &arena, MPOL_BIND, fastb->maskp));
	/* allocation */
	a = aml_area_malloc(&slow, memsize);
	b = aml_area_malloc(&slow, memsize);
	c = aml_area_malloc(&fast, memsize);
	assert(a != NULL && b != NULL && c != NULL);
	for(unsigned long i = 0; i < N*N; i++) {
		a[i] = (double)rand();
		b[i] = (double)rand();
		c[i] = 0.0;
	}
	clock_gettime(CLOCK_REALTIME, &start);
	do_work();
	clock_gettime(CLOCK_REALTIME, &stop);
	long long int time = 0;
	time =  (stop.tv_nsec - start.tv_nsec) +
                1e9* (stop.tv_sec - start.tv_sec);
	double flops = (2.0*N*N*N)/(time/1e9);
	/* print the flops in GFLOPS */
	printf("dgemm-mkl: %llu %lld %lld %f\n", N, memsize, time, flops/1e9);
	aml_area_free(&slow, a);
	aml_area_free(&slow, b);
	aml_area_free(&fast, c);
	aml_area_linux_destroy(&slow);
	aml_area_linux_destroy(&fast);
	aml_tiling_destroy(&tiling, AML_TILING_TYPE_2D_CONTIG);
	aml_finalize();
	return 0;
}
