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

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>

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
	   double *at);
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
	   double *at);
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
	   double *at);
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
	   double *at);
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
	   double *at);
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
	  double *at);
// To Do
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
// To Do
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
// To Do
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
// To Do
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

// To Do
void dtbmv(char uplo, char trans, char diag, size_t n, size_t k, double *a,
	   double *x);
// To Do
void dtbsv(char uplo, char trans, char diag, size_t n, size_t k, double *a,
	   double *x);
// To Do
void dtpmv(char uplo, char trans, char diag, size_t n, double *a, double *x);
// To Do
void dtpsv(char uplo, char trans, char diag, size_t n, double *a, double *x);
// To Do
void dtrmv(char uplo, char trans, char diag, size_t n, double *a, double *x);

