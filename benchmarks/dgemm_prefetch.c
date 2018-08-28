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

AML_TILING_2D_CONTIG_ROWMAJOR_DECL(tiling_row);
AML_TILING_2D_CONTIG_COLMAJOR_DECL(tiling_col);
AML_TILING_1D_DECL(tiling_prefetch);
AML_AREA_LINUX_DECL(slow);
AML_AREA_LINUX_DECL(fast);
AML_SCRATCH_PAR_DECL(sa);
AML_SCRATCH_PAR_DECL(sb);

size_t memsize, tilesize, N, T;
double *a, *b, *c;
struct timespec start, stop;

void do_work()
{
	int lda = (int)T, ldb, ldc;
	ldb = lda;
	ldc = lda;
	size_t ndims[2];
	double *prea, *preb;
	int ai, bi, oldai, oldbi;
	void *abaseptr, *bbaseptr;
	struct aml_scratch_request *ar, *br;
	aml_tiling_ndims(&tiling_row, &ndims[0], &ndims[1]);
	abaseptr = aml_scratch_baseptr(&sa);
	bbaseptr = aml_scratch_baseptr(&sb);
	prea = aml_tiling_tilestart(&tiling_prefetch, a, 0);
	preb = aml_tiling_tilestart(&tiling_prefetch, b, 0);
	ai = -1; bi = -1;

	for(int k = 0; k < ndims[1]; k++)
	{
		oldbi = bi;
		oldai = ai;
		aml_scratch_async_pull(&sa, &ar, abaseptr, &ai, a, k + 1);
		aml_scratch_async_pull(&sb, &br, bbaseptr, &bi, b, k + 1);
		#pragma omp parallel for collapse(2)
		for(int i = 0; i < ndims[0]; i++)
		{
			for(int j = 0; j < ndims[1]; j++)
			{
				size_t coff;
				double *ap, *bp, *cp;
				ap = aml_tiling_tilestart(&tiling_row, prea, i);
				bp = aml_tiling_tilestart(&tiling_row, preb, j);
				coff = i*ndims[1] + j;
				cp = aml_tiling_tilestart(&tiling_row, c, coff);
				cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ldc, lda, ldb, 1.0, ap, lda, bp, ldb, 1.0, cp, ldc);
			}
		}
		aml_scratch_wait(&sa, ar);
		aml_scratch_wait(&sb, br);
		prea = aml_tiling_tilestart(&tiling_prefetch, abaseptr, ai);
		preb = aml_tiling_tilestart(&tiling_prefetch, bbaseptr, bi);
		aml_scratch_release(&sa, oldai);
		aml_scratch_release(&sb, oldbi);
	}
}

int main(int argc, char* argv[])
{
	AML_ARENA_JEMALLOC_DECL(arns);
	AML_ARENA_JEMALLOC_DECL(arnf);
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

	/* the initial tiling, 2d grid of tiles */
	assert(!aml_tiling_init(&tiling_row, AML_TILING_TYPE_2D_CONTIG_ROWMAJOR,
				tilesize, memsize, N/T , N/T));
	assert(!aml_tiling_init(&tiling_col, AML_TILING_TYPE_2D_CONTIG_COLMAJOR,
				tilesize, memsize, N/T , N/T));
	/* the prefetch tiling, 1D sequence of columns of tiles */
	assert(!aml_tiling_init(&tiling_prefetch, AML_TILING_TYPE_1D,
				tilesize*(N/T), memsize));

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
	assert(!aml_dma_linux_seq_init(&dma, 2));
	assert(!aml_scratch_par_init(&sa, &fast, &slow, &dma, &tiling_prefetch, (size_t)2, (size_t)2));
	assert(!aml_scratch_par_init(&sb, &fast, &slow, &dma, &tiling_prefetch, (size_t)2, (size_t)2));
	/* allocation */
	a = aml_area_malloc(&slow, memsize);
	b = aml_area_malloc(&slow, memsize);
	c = aml_area_malloc(&fast, memsize);
	assert(a != NULL && b != NULL && c != NULL);

	size_t ntilerows, ntilecols, tilerowsize, tilecolsize, rowsize, colsize;
	rowsize = colsize = N;
	tilerowsize = tilecolsize = T;
	ntilerows = ntilecols = N/T;
	for(unsigned long i = 0; i < N*N; i+=tilerowsize) {
		size_t tilerow, tilecol, row, column;
		/* Tile row index (row-major).  */
		tilerow = i / (tilerowsize * tilecolsize * ntilerows);
		/* Tile column index (row-major).  */
		tilecol = (i / tilerowsize) % ntilerows;
		/* Row index within a tile (row-major).  */
		row = (i / rowsize) % tilecolsize;
		/* Column index within a tile (row-major).  */
		/* column = i % tilerowsize; */

		size_t a_offset, b_offset;
		/* Tiles in A need to be transposed (column-major).  */
		a_offset = (tilecol * ntilecols + tilerow) *
			tilerowsize * tilecolsize +
			row * tilerowsize;
		/* Tiles in B are in row-major order.  */
		b_offset = (tilerow * ntilerows + tilecol) *
			tilerowsize * tilecolsize +
			row * tilerowsize;
		for (column = 0; column < tilerowsize; column++) {
			a[a_offset + column] = (double)rand();
			b[b_offset + column] = (double)rand();
			/* C is tiled as well (row-major) but since it's
			   all-zeros at this point, we don't bother.  */
			c[i+column] = 0.0;
		}
	}

	clock_gettime(CLOCK_REALTIME, &start);
	do_work();
	clock_gettime(CLOCK_REALTIME, &stop);
	long long int time = 0;
	time =  (stop.tv_nsec - start.tv_nsec) +
                1e9* (stop.tv_sec - start.tv_sec);
	double flops = (2.0*N*N*N)/(time/1e9);

	/* De-tile the result matrix (C).  I couldn't figure out how to do
	   it in-place so we are de-tiling to the A matrix.  */
	for(unsigned long i = 0; i < N*N; i+=tilerowsize) {
		size_t tilerow, tilecol, row;
		/* Tile row index (row-major).  */
		tilerow = i / (tilerowsize * tilecolsize * ntilerows);
		/* Tile column index (row-major).  */
		tilecol = (i / tilerowsize) % ntilerows;
		/* Row index within a tile (row-major).  */
		row = (i / rowsize) % tilecolsize;
		/* i converted to tiled.  */
		unsigned long tiledi = (tilerow * ntilerows + tilecol) *
			tilerowsize * tilecolsize + row * tilerowsize;

		memcpy(&a[i], &c[tiledi], tilerowsize*sizeof(double));
	}

	/* print the flops in GFLOPS */
	printf("dgemm-prefetch: %llu %lld %lld %f\n", N, memsize, time,
	       flops/1e9);
	aml_scratch_par_destroy(&sa);
	aml_scratch_par_destroy(&sb);
	aml_dma_linux_seq_destroy(&dma);
	aml_area_free(&slow, a);
	aml_area_free(&slow, b);
	aml_area_free(&fast, c);
	aml_area_linux_destroy(&slow);
	aml_area_linux_destroy(&fast);
	aml_tiling_destroy(&tiling_row, AML_TILING_TYPE_2D_CONTIG_ROWMAJOR);
	aml_tiling_destroy(&tiling_col, AML_TILING_TYPE_2D_CONTIG_ROWMAJOR);
	aml_tiling_destroy(&tiling_prefetch, AML_TILING_TYPE_1D);
	aml_finalize();
	return 0;
}