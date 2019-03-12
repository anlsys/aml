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

#define N 2
#define M 3

int main(int argc, char *argv[])
{
	AML_TILING_2D_ROWMAJOR_DECL(trm);
	AML_TILING_2D_ROWMAJOR_DECL(trt);
	AML_TILING_2D_COLMAJOR_DECL(tcm);
	AML_TILING_2D_ROWMAJOR_DECL(tct);

	/* Matrices used for checks:
	 *  - rowm: stored in row-major, numbered in memory order
	 *  - rowt: stored in row-major, transposition of rowm
	 *  - colm: stored in col-major, number in memory order
	 *  - colt: stored in col-major, transposition of colt
	 *
	 * Matrices shapes:
	 *
	 *      rowm/colt                rowt/colm
	 *    +---+---+---+              +---+---+
	 *    | 0 | 1 | 2 |              | 0 | 3 |
	 *    +---+---+---+              +---+---+
	 *    | 3 | 4 | 5 |              | 1 | 4 |
	 *    +---+---+---+              +---+---+
	 *                               | 2 | 5 |
	 *                               +---+---+
	 * Matrices in memory:
	 *
	 *      rowm/colm                    rowt/colt 
	 *   +---+---+---+---+---+      +---+---+---+---+---+---+
	 *   | 0 | 1 | 2 | 4 | 5 |      | 0 | 3 | 1 | 4 | 2 | 5 |
	 *   +---+---+---+---+---+      +---+---+---+---+---+---+
	 */
	int num;
	int rowm[N][M];
	int rowt[M][N];
	int colm[M][N];
	int colt[N][M];

	/* library initialization */
	aml_init(&argc, &argv);

	/* init matrices */
	for(int i = 0; i < N*M; i++)
		((int*)rowm)[i] = i;
	for(int i = 0; i < M; i++)
		for(int j = 0; j < N; j++)
			rowt[i][j] = rowm[j][i];
	memcpy(colm, rowm, N*M*sizeof(int));
	memcpy(colt, rowt, N*M*sizeof(int));

	/* initialize the tilings */
	aml_tiling_init(&trm, AML_TILING_TYPE_2D_ROWMAJOR,
			sizeof(int), N*M*sizeof(int), N, M);
	aml_tiling_init(&trt, AML_TILING_TYPE_2D_ROWMAJOR,
			sizeof(int), N*M*sizeof(int), M, N);
	aml_tiling_init(&tcm, AML_TILING_TYPE_2D_COLMAJOR,
			sizeof(int), N*M*sizeof(int), M, N);
	aml_tiling_init(&tct, AML_TILING_TYPE_2D_COLMAJOR,
			sizeof(int), N*M*sizeof(int), N, M);

	size_t ndims[2];
	aml_tiling_ndims(&trm, &ndims[0], &ndims[1]);
	assert(ndims[0] == N && ndims[1] == M);
	aml_tiling_ndims(&tcm, &ndims[0], &ndims[1]);
	assert(ndims[0] == M && ndims[1] == N);

	/* check that the tilings gives me the right ids */
	num = 0;
	for(int i = 0; i < N; i++)
		for(int j = 0; j < M; j++)
		{
			int irow = aml_tiling_tileid(&trm, i, j);
			int icol = aml_tiling_tileid(&tcm, j, i);
			assert(irow == icol && irow == num);
			num++;
		}

	/* check that the tiling gives the right starts */
	num = 0;
	for(int i = 0; i < N; i++)
		for(int j = 0; j < M; j++)
		{
			int irow = aml_tiling_tileid(&trm, i, j);
			int icol = aml_tiling_tileid(&tcm, j, i);
			int *rm = aml_tiling_tilestart(&trm, &rowm, irow);
			int *cm = aml_tiling_tilestart(&tcm, &colm, icol);
			assert(*rm == num && *cm == num);
			num++;
		}

	/* check that applying a column-major tiling on a row-major matrix gives
	 * us its transposition. */
	for(int i = 0; i < N; i++)
		for(int j = 0; j < M; j++)
		{
			int icm = aml_tiling_tileid(&tcm, j, i);
			int *cm = aml_tiling_tilestart(&tcm, &rowm, icm);
			int irt = aml_tiling_tileid(&trt, j, i);
			int *rt = aml_tiling_tilestart(&trt, &rowt, irt);
			assert(*cm == *rt);
		}

	/* check that applying a row-major tiling on a col-major matrix gives
	 * us its transposition. */
	for(int i = 0; i < M; i++)
		for(int j = 0; j < N; j++)
		{
			int irm = aml_tiling_tileid(&trm, j, i);
			int *rm = aml_tiling_tilestart(&trm, &colm, irm);
			int ict = aml_tiling_tileid(&tct, j, i);
			int *ct = aml_tiling_tilestart(&tct, &rowt, ict);
			assert(*rm == *ct);
		}


	/* delete the tilings */
	aml_tiling_destroy(&trm, AML_TILING_TYPE_2D_ROWMAJOR);
	aml_tiling_destroy(&trt, AML_TILING_TYPE_2D_ROWMAJOR);
	aml_tiling_destroy(&tcm, AML_TILING_TYPE_2D_COLMAJOR);
	aml_tiling_destroy(&tct, AML_TILING_TYPE_2D_COLMAJOR);
	aml_finalize();
	return 0;
}
