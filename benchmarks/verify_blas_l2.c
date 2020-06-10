/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include "verify_blas_l2.h"

void init_matrix_mn(size_t m,
                    size_t n,
                    int kl,
                    int ku,
                    double **a,
                    double *x,
                    double *y,
                    double *at)
{
	size_t i, j;

#pragma omp parallel for
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			if (((j >= i) && (j < i + kl)) ||
			    ((i >= j) && (i > j + ku))) {
				a[i][j] = (double)(i + j);
				at[i * m + j] = (double)(i + j);
			} else {
				a[i][j] = 0.0;
				at[i * m + j] = 0.0;
			}
		}
		y[i] = (double)(m - i);
	}

#pragma omp parallel for
	for (i = 0; i < n; i++) {
		x[i] = (double)i;
	}
}

void init_matrix_n(
        size_t n, int kl, int ku, double **a, double *x, double *y, double *at)
{
	size_t i, j;

#pragma omp parallel for
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			if (((j >= i) && (j < i + kl)) ||
			    ((i >= j) && (i > j + ku))) {
				a[i][j] = (double)(i + j);
				at[i * n + j] = (double)(i + j);
			} else {
				a[i][j] = 0.0;
				at[i * n + j] = 0.0;
			}
		}
		y[i] = (double)(n - i);
		x[i] = (double)i;
	}
}

// TODO Implement all those functions
int verify_dgbmv(size_t m,
                 size_t n,
                 int kl,
                 int ku,
                 double alpha,
                 double beta,
                 double **a,
                 double *x,
                 double *y,
                 double *at)
{
	(void)*at;

	//	size_t i, j;

	(void)m;
	(void)n;
	(void)kl;
	(void)ku;
	(void)alpha;
	(void)beta;
	(void)**a;
	(void)*x;
	(void)*y;
}

int verify_dgemv(size_t m,
                 size_t n,
                 int kl,
                 int ku,
                 double alpha,
                 double beta,
                 double **a,
                 double *x,
                 double *y,
                 double *at)
{
	(void)kl;
	(void)ku;
	(void)*at;

	(void)m;
	(void)n;
	(void)alpha;
	(void)beta;
	(void)**a;
	(void)*x;
	(void)*y;
}

int verify_dger(size_t m,
                size_t n,
                int kl,
                int ku,
                double alpha,
                double beta,
                double **a,
                double *x,
                double *y,
                double *at)
{
	(void)kl;
	(void)ku;
	(void)beta;
	(void)*at;

	(void)m;
	(void)n;
	(void)alpha;
	(void)**a;
	(void)*x;
	(void)*y;
}

int verify_dsbmv(size_t m,
                 size_t n,
                 int kl,
                 int ku,
                 double alpha,
                 double beta,
                 double **a,
                 double *x,
                 double *y,
                 double *at)
{
	(void)m;
	(void)ku;
	(void)*at;

	(void)n;
	(void)kl;
	(void)alpha;
	(void)beta;
	(void)**a;
	(void)*x;
	(void)*y;
}

int verify_dspmv(size_t m,
                 size_t n,
                 int kl,
                 int ku,
                 double alpha,
                 double beta,
                 double **a,
                 double *x,
                 double *y,
                 double *at)
{
	(void)m;
	(void)kl;
	(void)ku;
	(void)**a;

	(void)n;
	(void)alpha;
	(void)beta;
	(void)*x;
	(void)*y;
	(void)*at;
}

int verify_dspr(size_t m,
                size_t n,
                int kl,
                int ku,
                double alpha,
                double beta,
                double **a,
                double *x,
                double *y,
                double *at)
{
	(void)m;
	(void)kl;
	(void)ku;
	(void)beta;
	(void)**a;
	(void)*y;

	(void)n;
	(void)alpha;
	(void)*x;
	(void)*at;
}

int verify_dspr2(size_t m,
                 size_t n,
                 int kl,
                 int ku,
                 double alpha,
                 double beta,
                 double **a,
                 double *x,
                 double *y,
                 double *at)
{
	(void)m;
	(void)kl;
	(void)ku;
	(void)beta;
	(void)**a;

	(void)n;
	(void)alpha;
	(void)*x;
	(void)*y;
	(void)*at;
}

int verify_dsymv(size_t m,
                 size_t n,
                 int kl,
                 int ku,
                 double alpha,
                 double beta,
                 double **a,
                 double *x,
                 double *y,
                 double *at)
{
	(void)m;
	(void)kl;
	(void)ku;
	(void)*at;

	(void)n;
	(void)alpha;
	(void)beta;
	(void)**a;
	(void)*x;
	(void)*y;
}

int verify_dsyr(size_t m,
                size_t n,
                int kl,
                int ku,
                double alpha,
                double beta,
                double **a,
                double *x,
                double *y,
                double *at)
{
	(void)m;
	(void)kl;
	(void)ku;
	(void)beta;
	(void)*y;
	(void)*at;

	(void)n;
	(void)alpha;
	(void)**a;
	(void)*x;
}

int verify_dsyr2(size_t m,
                 size_t n,
                 int kl,
                 int ku,
                 double alpha,
                 double beta,
                 double **a,
                 double *x,
                 double *y,
                 double *at)
{
	(void)m;
	(void)kl;
	(void)ku;
	(void)beta;
	(void)*at;

	(void)n;
	(void)alpha;
	(void)**a;
	(void)*x;
	(void)*y;
}

int verify_dtbmv(size_t m,
                 size_t n,
                 int kl,
                 int ku,
                 double alpha,
                 double beta,
                 double **a,
                 double *x,
                 double *y,
                 double *at)
{
	(void)m;
	(void)ku;
	(void)alpha;
	(void)beta;
	(void)*y;
	(void)*at;

	(void)n;
	(void)kl;
	(void)**a;
	(void)*x;
}

int verify_dtbsv(size_t m,
                 size_t n,
                 int kl,
                 int ku,
                 double alpha,
                 double beta,
                 double **a,
                 double *x,
                 double *y,
                 double *at)
{
	(void)m;
	(void)ku;
	(void)alpha;
	(void)beta;
	(void)*y;
	(void)*at;

	(void)n;
	(void)kl;
	(void)**a;
	(void)*x;
}

int verify_dtpmv(size_t m,
                 size_t n,
                 int kl,
                 int ku,
                 double alpha,
                 double beta,
                 double **a,
                 double *x,
                 double *y,
                 double *at)
{
	(void)m;
	(void)kl;
	(void)ku;
	(void)alpha;
	(void)beta;
	(void)**a;
	(void)*y;

	(void)n;
	(void)*x;
	(void)*at;
}

int verify_dtpsv(size_t m,
                 size_t n,
                 int kl,
                 int ku,
                 double alpha,
                 double beta,
                 double **a,
                 double *x,
                 double *y,
                 double *at)
{
	(void)m;
	(void)kl;
	(void)ku;
	(void)alpha;
	(void)beta;
	(void)**a;
	(void)*y;

	(void)n;
	(void)*x;
	(void)*at;
}

int verify_dtrmv(size_t m,
                 size_t n,
                 int kl,
                 int ku,
                 double alpha,
                 double beta,
                 double **a,
                 double *x,
                 double *y,
                 double *at)
{
	(void)m;
	(void)kl;
	(void)ku;
	(void)alpha;
	(void)beta;
	(void)*y;
	(void)*at;

	(void)n;
	(void)**a;
	(void)*x;
}
