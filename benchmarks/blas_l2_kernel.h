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

double dgbmv(bool trans,
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

double dgemv(bool trans,
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

double dger(bool trans,
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

double dsbmv(bool trans,
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

double dspmv(bool trans,
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

double dspr(bool trans,
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

double dspr2(bool trans,
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

double dsymv(bool trans,
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

double dsyr(bool trans,
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

double dsyr2(bool trans,
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

double dtbmv(bool trans,
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

double dtbsv(bool trans,
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

double dtpmv(bool trans,
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

double dtpsv(bool trans,
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

double dtrmv(bool trans,
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
