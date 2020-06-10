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

void init_matrix_mn(size_t m,
                    size_t n,
                    int kl,
                    int ku,
                    double **a,
                    double *x,
                    double *y,
                    double *at);

void init_matrix_n(
        size_t n, int kl, int ku, double **a, double *x, double *y, double *at);

typedef int (*v)(size_t,
                 size_t,
                 int,
                 int,
                 double,
                 double,
                 double **,
                 double *,
                 double *,
                 double *);

int verify_dgbmv(size_t m,
                 size_t n,
                 int kl,
                 int ku,
                 double alpha,
                 double beta,
                 double **a,
                 double *x,
                 double *y,
                 double *at);

int verify_dgemv(size_t m,
                 size_t n,
                 int kl,
                 int ku,
                 double alpha,
                 double beta,
                 double **a,
                 double *x,
                 double *y,
                 double *at);

int verify_dger(size_t m,
                size_t n,
                int kl,
                int ku,
                double alpha,
                double beta,
                double **a,
                double *x,
                double *y,
                double *at);

int verify_dsbmv(size_t m,
                 size_t n,
                 int kl,
                 int ku,
                 double alpha,
                 double beta,
                 double **a,
                 double *x,
                 double *y,
                 double *at);

int verify_dspmv(size_t m,
                 size_t n,
                 int kl,
                 int ku,
                 double alpha,
                 double beta,
                 double **a,
                 double *x,
                 double *y,
                 double *at);

int verify_dspr(size_t m,
                size_t n,
                int kl,
                int ku,
                double alpha,
                double beta,
                double **a,
                double *x,
                double *y,
                double *at);

int verify_dspr2(size_t m,
                 size_t n,
                 int kl,
                 int ku,
                 double alpha,
                 double beta,
                 double **a,
                 double *x,
                 double *y,
                 double *at);

int verify_dsymv(size_t m,
                 size_t n,
                 int kl,
                 int ku,
                 double alpha,
                 double beta,
                 double **a,
                 double *x,
                 double *y,
                 double *at);

int verify_dsyr(size_t m,
                size_t n,
                int kl,
                int ku,
                double alpha,
                double beta,
                double **a,
                double *x,
                double *y,
                double *at);

int verify_dsyr2(size_t m,
                 size_t n,
                 int kl,
                 int ku,
                 double alpha,
                 double beta,
                 double **a,
                 double *x,
                 double *y,
                 double *at);

int verify_dtbmv(size_t m,
                 size_t n,
                 int kl,
                 int ku,
                 double alpha,
                 double beta,
                 double **a,
                 double *x,
                 double *y,
                 double *at);

int verify_dtbsv(size_t m,
                 size_t n,
                 int kl,
                 int ku,
                 double alpha,
                 double beta,
                 double **a,
                 double *x,
                 double *y,
                 double *at);

int verify_dtpmv(size_t m,
                 size_t n,
                 int kl,
                 int ku,
                 double alpha,
                 double beta,
                 double **a,
                 double *x,
                 double *y,
                 double *at);

int verify_dtpsv(size_t m,
                 size_t n,
                 int kl,
                 int ku,
                 double alpha,
                 double beta,
                 double **a,
                 double *x,
                 double *y,
                 double *at);

int verify_dtrmv(size_t m,
                 size_t n,
                 int kl,
                 int ku,
                 double alpha,
                 double beta,
                 double **a,
                 double *x,
                 double *y,
                 double *at);
