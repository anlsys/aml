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
AML_AREA_LINUX_DECL(slow);
AML_AREA_LINUX_DECL(fast);

size_t memsize, tilesize, N, T;
double *a, *b, *c;
//globalOffset will give us the Offset for which tile of C is being computed 
//iterOffset will give us the offset of which tiles of A and B are being used
unsigned long aGlobalOffset, bGlobalOffset, cGlobalOffset, numThreads;
struct timespec start, stop;

void do_work()
{
	int lda = (int)T, ldb, ldc;
	ldb = lda;
	ldc = lda;
	size_t ndims[2];
	ndims[0] = (size_t)sqrt(numThreads);
	ndims[1] = (size_t)sqrt(numThreads);

	//This for loop will assign each thread a specific C tile in the matrix like so:
	// (4x4 C matrix with 16 threads, numbers are which tid gets the tile)
	//
	// 0 	1	2 	3
	// 4 	5 	6 	7
	// 8 	9 	10 	11
	// 12	13	14	15
	//
	//This method will be more cache efficient and hopefully will help prefetching and L2 hit rate
	#pragma omp parallel for
	for(int k = 0; k < ndims[0] * ndims[1]; k++){
		int bCol = k % ndims[0];
		int aRow = k / ndims[1];
		for (int i = 0; i < ndims[0]; i++){
			size_t aoff, boff, coff;
			double *ap, *bp, *cp;
			aoff = i + (aRow * ndims[1]) + aGlobalOffset;
			boff = (i * ndims[0]) + bGlobalOffset;
			coff = k + cGlobalOffset;
			ap = aml_tiling_tilestart(&tiling_col, a, aoff);
			bp = aml_tiling_tilestart(&tiling_row, b, boff);
			cp = aml_tiling_tilestart(&tiling_row, c, coff);
			cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ldc, lda, ldb, 1.0, ap, lda, bp, ldb, 1.0, cp, ldc);
		}
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
	numThreads = omp_get_max_threads();

	printf("Memsize: %lu\nTilesize: %lu\nNum Threads: %d\n", memsize, tilesize, numThreads);
	/* the initial tiling, of 2D square tiles */
	assert(!aml_tiling_init(&tiling_row, AML_TILING_TYPE_2D_CONTIG_ROWMAJOR,
				tilesize, memsize, N/T , N/T));
	assert(!aml_tiling_init(&tiling_col, AML_TILING_TYPE_2D_CONTIG_COLMAJOR,
				tilesize, memsize, N/T , N/T));

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
	
	//This is hardcoded to split the entire matrix into chunks that have tiles equal to number of threads currently.
	//This is also going to assume that the matrices are currently properly transformed to match whatever is needed.
	//Formatting the matrix should not affect # of floating ops or locality in anyway. (But checking may be a good idea)
	//printf("There are %d tiles in this matrix\n", ntilerows * ntilecols);

	//This loop will iterate through the macro-tiles of A in a row. At the end of the loop, the entire matrix is done
	for(unsigned long i = 0; i < ntilerows; i += (unsigned long)sqrt(numThreads)){
		//This loop will iterate through macro-tiles of A in a col. At the end of the loop the entire macro-Column of A and respective C will never be needed again.
		for(unsigned long j = 0; j < ntilecols; j += (unsigned long)sqrt(numThreads)){
			aGlobalOffset = (j * ntilerows) + i; // (How many rows to offset in normal tiles) + how many cols to offset in normal tiles  
			//printf("Dealing with macro-tile A %lu of %lu in macro-column %lu\n", j/(unsigned long)sqrt(numThreads) + 1, ntilerows / (unsigned long)sqrt(numThreads), i/(unsigned long)sqrt(numThreads));
			//This loop will iterate through the macro-tiles of B in a row. At the end of the inner loop
			//the A tile being used will not need to be used ever again.
			for(unsigned long k = 0; k < ntilerows; k += (unsigned long)sqrt(numThreads)){
				bGlobalOffset = k + i*ntilerows; //(which normal tile in row is beginning of macro-tile) + (How may rows to offset)
				cGlobalOffset = (j * ntilerows) + k; //(which row) + (which column) 
				do_work();
			}
		}
		
	}

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
	printf("dgemm-noprefetch: %llu %lld %lld %lf\n", N, memsize, time,
	       flops/1e9);
	aml_area_free(&slow, a);
	aml_area_free(&slow, b);
	aml_area_free(&fast, c);
	aml_area_linux_destroy(&slow);
	aml_area_linux_destroy(&fast);
	aml_tiling_destroy(&tiling_row, AML_TILING_TYPE_2D_CONTIG_ROWMAJOR);
	aml_tiling_destroy(&tiling_col, AML_TILING_TYPE_2D_CONTIG_ROWMAJOR);
	aml_finalize();
	return 0;
}
