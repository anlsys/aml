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
#include <stdbool.h>
#include <stdlib.h>
#include <unistd.h>

void dgbmv(bool trans,
           bool uplo,
           bool unit,
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
           bool uplo,
           bool unit,
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
          bool uplo,
          bool unit,
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
           bool uplo,
           bool unit,
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
           bool uplo,
           bool unit,
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
          bool uplo,
          bool unit,
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

void dspr2(bool trans,
           bool uplo,
           bool unit,
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

void dsymv(bool trans,
           bool uplo,
           bool unit,
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

void dsyr(bool trans,
          bool uplo,
          bool unit,
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

void dsyr2(bool trans,
           bool uplo,
           bool unit,
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

void dtbmv(bool trans,
           bool uplo,
           bool unit,
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
void dtbsv(bool trans,
           bool uplo,
           bool unit,
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
void dtpmv(bool trans,
           bool uplo,
           bool unit,
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
void dtpsv(bool trans,
           bool uplo,
           bool unit,
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
void dtrmv(bool trans,
           bool uplo,
           bool unit,
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
