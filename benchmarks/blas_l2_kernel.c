/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *******************************************************************************/
/*
 * This is a benchmark for the BLAS Level 2 operations for AML.
 */

#include "blas_l1_kernel.h"
#include "blas_l2_kernel.h"

/* Look into another way to define these */
#define sign(a) ((a > 0) ? 1 : ((a < 0) ? -1 : 0))

void dgbmv(bool trans,
	   char uplo,
	   size_t m,
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
	size_t i, j, kup, k;
	double temp;
	if (m == 0 || n == 0 || (alpha == 0 && beta == 1))
		return;
	k = trans ? m : n;

	// y = beta * y
	if (beta != 1) {
		if (beta == 0) {
			for (i = 0; i < k; i++)
				y[i] = 0;
		}
		else {
			for (i = 0; i < k; i++)
				y[i] = beta *y[i];
		}
	}
	if (alpha == 0)
		return;
	kup = ku + 1;
	if (!trans) {
		// y = alpha * A * x + y
		for (j = 0; j < n; j ++) {
			temp = alpha * x[j];
			k = kup - j;
			for (i = fmax(0, j - ku); i < fmin(m, j + kl); i++)
				y[i] += temp * a[k+i][j];
		}
	}
	else {
		// y = alpha * A * T * x + y
		for (j = 0; j < n; j ++) {
			temp = 0;
			for (i = fmax(0, j - ku); i < fmin(m, j + kl); i++)
				temp += a[k+i][j] * x[i];
			y[j] += alpha * temp;
		}
	}
}

/* y = alpha * a * x + beta * y if trans = 0, y = alpha * a^T * x + beta * y if
 * trans = 1 or c */
void dgemv(bool trans,
	   char uplo,
	   size_t m,
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
	size_t i, j;

	if (beta != 1) {
		i = trans ? n : m;
		dscal(i, beta, y, y);
	} 
	if (alpha == 0)
		return;
	else {
		if (trans) {
			for (j = 0; j < m; j ++){
				for (i = 0; i < n; i++)
					y[i] += alpha * x[j] * a[j][i];
			}
		} else {
			for (i = 0; i < m; i++)
				y[i] += alpha * ddot(n, a[i], x);
		}
	}
}

/* a = a + alpha * x * y^T */
void dger(bool trans,
	  char uplo,
	  size_t m,
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
	size_t i,j;
	if (m == 0 || n == 0 || alpha == 0)
		return;
	else {
		for (j = 0; j < n; j++) {
			for (i = 0; i <m; i++) 
				a[i][j] += alpha * y[j] * x[i];
		}
	}
}

// y = alpha * A * x + beta * y, A being a n by symmetric matrix, with k
// super-diagonals
void dsbmv(bool trans,
	   char uplo,
	   size_t m,
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
	size_t i, j;
	int l;
	double temp, temp2;

	if (n == 0 || (alpha == 0 && beta == 1))
		return;

	// y = beta * y
	if (beta != 1) {
		if (beta == 0) {
			for (i = 0; i < n; i++)
				y[i] = 0;
		} 
		else {
			for (i = 0; i < n; i++)
				y[i] = beta * y[i];
		}
	}
	if (alpha == 0)
		return;
	// upper triangle
	if (uplo = 'u') {
		int kp1 = kl + 1;
		for (j = 0; j < n; j ++) {
			temp = alpha * x[j];
			temp2 = 0.0;
			l = kp1 -j;
			for (i = fmax(1, j-kl); i < j-1; i++) {
				y[i] += temp * a[l+i][j];
				temp2 += a[l+i][j] * x[i];
			}
			y[j] += temp * a[kp1][j] + alpha * temp2;
		}
	}
	// lower triangle
	else {
		for (j = 0; j < n; j ++) {
			temp = alpha * x[j];
			temp2 = 0.0;
			y[j] += temp * a[1][j];
			l = 1 -j;
			for (i = j + 1; i < fmin(n, j+kl); i++) {
				y[i] += temp * a[l+i][j];
				temp2 += a[l+i][j] * x[i];
			}
			y[j] += alpha * temp2;
		}
	}
}

/* y = alpha * a *x + beta * y, a symmetric matrix, uplo meaning upper or lower
 * triangular (u or l) */
void dspmv(bool trans,
	   char uplo,
	   size_t m,
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
	size_t i, j, kk, k;
        double temp, temp2;

	if (n == 0 || (alpha == 0 && beta == 1)) 
		return;

	// y = beta * y
	if (beta != 1) {
		if (beta == 0) {
			for (i = 0; i < n; i++)
				y[i] = 0;
		} else {
			for (i = 0; i < n; i++)
				y[i] = beta * y[i];
		}
	}
	if (alpha == 0)
		return;
	// y = alpha * x + beta * y
	kk = 0;
	if (uplo == 'u') {
		// upper triangle
		for (i = 0; i < n; i++) {
			temp = alpha * x[i];
			temp2 = 0.0;
			k = kk;
			for (j = 0; j < i; j ++) {
				y[j] += temp * at[k];
				temp2 += at[k] * x[j];
				k++;
			}
			y[i] += temp * at[kk+i-1] + alpha * temp2;
			kk += j;
		}
	} else {
		for (i = 0; i < n; i++) {
			temp = alpha * x[i];
			temp2 = 0.0;
			y[i] += temp * at[kk];
			k = kk + 1;
			for (j = i + 1; j < n; j++) {
				y[j] += temp * at[k];
				temp2 += at[k] * x[j];
				k++;
			}
			y[i] += alpha * temp2;
			kk = kk + (n -i);
		}
	}
        
}

/* A = alpha * x * x**T + A, A a n by n symmetric matrix, in packed form, uplo
 * meaning upper or lower triangular (u or l).
 */
void dspr(bool trans,
	  char uplo,
	  size_t m,
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
	size_t i, j, k, kk;
	double temp;

	if (n == 0 || alpha == 0)
		return;
	kk = 0;
	if (uplo == 'u') {
		// upper triangular form stored
		for (j = 0; j < n; j++) {
			if (x[j] != 0) {
				temp = alpha * x[j];
				k = kk;
				for (i = 0; i < n; i++) {
					at[k] += at[k] + x[i] * temp;
					k++;
				}
			}
			kk += j;
		}
	}
	else {
		for (j = 0; j < n; j++) {
			if (x[j] != 0) {
				temp = alpha * x[j];
				k = kk;
				for (i = 0; i < n; i++) {
					at[k] += x[i] * temp;
					k++;
				}
			}
			kk += n - j + 1;
		}
	}
}

void dspr2(bool trans,
	   char uplo,
	   size_t m,
	   size_t n,
	   int kl,
	   int ku,
	   double alpha,
	   double beta,
	   double **a,
	   double *x,
	   double *y,
	   double *at);
{}

void dsymv(bool trans,
	   char uplo,
	   size_t m,
	   size_t n,
	   int kl,
	   int ku,
	   double alpha,
	   double beta,
	   double **a,
	   double *x,
	   double *y,
	   double *at);
{}

void dsyr(bool trans,
	  char uplo,
	  size_t m,
	  size_t n,
	  int kl,
	  int ku,
	  double alpha,
	  double beta,
	  double **a,
	  double *x,
	  double *y,
	  double *at);
{}

void dsyr2(bool trans,
	   char uplo,
	   size_t m,
	   size_t n,
	   int kl,
	   int ku,
	   double alpha,
	   double beta,
	   double **a,
	   double *x,
	   double *y,
	   double *at);
{}

// TODO: change the arguments to match the others (see char trans and add char
// diag)
void dtbmv(char uplo, char trans, char diag, size_t n, size_t k, double *a,
	   double *x)
{}

void dtbsv(char uplo, char trans, char diag, size_t n, size_t k, double *a,
	   double *x)
{}

void dtpmv(char uplo, char trans, char diag, size_t n, double *a, double *x)
{}

void dtpsv(char uplo, char trans, char diag, size_t n, double *a, double *x)
{}

void dtrmv(char uplo, char trans, char diag, size_t n, double *a, double *x)
{}

