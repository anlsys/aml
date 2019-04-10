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
#include "aml/tiling/2d.h"
#include <assert.h>
#include <errno.h>
#include <mkl.h>
#include <omp.h>
#include <pthread.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>

AML_TILING_2D_ROWMAJOR_DECL(tiling_row);
AML_TILING_2D_COLMAJOR_DECL(tiling_col);
struct aml_area *slow, *fast;
size_t memsize, tilesize, N, T;
double *a, *b, *c;
struct timespec start, stop;

void do_work()
{
	int lda = (int)T, ldb, ldc;
	ldb = lda;
	ldc = lda;
	size_t ndims[2];
	aml_tiling_ndims(&tiling_row, &ndims[0], &ndims[1]);

	for(int k = 0; k < ndims[1]; k++)
	{
		#pragma omp parallel for
		for(int i = 0; i < ndims[0]; i++)
		{
			for(int j = 0; j < ndims[1]; j++)
			{
				size_t aoff, boff, coff;
				double *ap, *bp, *cp;
				aoff = aml_tiling_tileid(&tiling_col, i, k);
				boff = aml_tiling_tileid(&tiling_row, k, j);
				coff = aml_tiling_tileid(&tiling_row, i, j);
				ap = aml_tiling_tilestart(&tiling_col, a, aoff);
				bp = aml_tiling_tilestart(&tiling_row, b, boff);
				cp = aml_tiling_tilestart(&tiling_row, c, coff);
				cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ldc, lda, ldb, 1.0, ap, lda, bp, ldb, 1.0, cp, ldc);
			}
		}
	}
}

int main(int argc, char* argv[])
{
	struct aml_bitmap slowb, fastb;
	aml_init(&argc, &argv);
	assert(argc == 5);
	assert(aml_bitmap_from_string(&fastb, argv[1]) == 0);
	assert(aml_bitmap_from_string(&slowb, argv[2]) == 0);
	N = atol(argv[3]);
	T = atol(argv[4]);
	/* let's not handle messy tile sizes */
	assert(N % T == 0);
	memsize = sizeof(double)*N*N;
	tilesize = sizeof(double)*T*T;

	/* the initial tiling, of 2D square tiles */
	assert(!aml_tiling_2d_init(&tiling_row, AML_TILING_TYPE_2D_ROWMAJOR,
				tilesize, memsize, N/T , N/T));
	assert(!aml_tiling_2d_init(&tiling_col, AML_TILING_TYPE_2D_COLMAJOR,
				tilesize, memsize, N/T , N/T));

	aml_area_linux_create(&slow, AML_AREA_LINUX_MMAP_FLAG_PRIVATE,
				     &slowb, AML_AREA_LINUX_BINDING_FLAG_BIND);
	assert(slow != NULL);
	aml_area_linux_create(&fast, AML_AREA_LINUX_MMAP_FLAG_PRIVATE,
				     &fastb, AML_AREA_LINUX_BINDING_FLAG_BIND);
	assert(fast != NULL);

	/* allocation */
	a = aml_area_mmap(slow, NULL, memsize);
	b = aml_area_mmap(slow, NULL, memsize);
	c = aml_area_mmap(fast, NULL, memsize);
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
	printf("dgemm-noprefetch: %llu %lld %lld %f\n", N, memsize, time,
	       flops/1e9);
	aml_area_munmap(slow, a, memsize);
	aml_area_munmap(slow, b, memsize);
	aml_area_munmap(fast, c, memsize);
	aml_area_linux_destroy(&slow);
	aml_area_linux_destroy(&fast);
	aml_tiling_2d_fini(&tiling_row);
	aml_tiling_2d_fini(&tiling_col);
	aml_finalize();
	return 0;
}
