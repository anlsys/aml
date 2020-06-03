/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include <assert.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

void init_arrays(size_t memsize, double *a, double *b, double *c);
void init_matrix(size_t m, size_t n, double **a);

// change the type
typedef int (*v)(size_t, double *, double *, double *, double, double);

void verify_dgbmv(bool trans,
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
void verify_dgemv(bool trans,
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
void verify_dger(bool trans,
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
void verify_dsbmv(bool trans,
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
void verify_dspmv(bool trans,
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
void verify_dspr(bool trans,
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
void verify_dspr2(bool trans,
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
void verify_dsymv(bool trans,
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
void verify_dsyr(bool trans,
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
void verify_dsyr2(bool trans,
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
