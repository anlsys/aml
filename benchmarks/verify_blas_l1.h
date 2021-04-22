/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include <assert.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

void init_arrays(size_t memsize, double *a, double *b, double *c);

typedef int (*v)(size_t, double *, double *, double *, double, double);
int verify_dasum(size_t memsize,
                 double *a,
                 double *b,
                 double *c,
                 double scalar,
                 double res);
int verify_daxpy(size_t memsize,
                 double *a,
                 double *b,
                 double *c,
                 double scalar,
                 double res);
int verify_dcopy(size_t memsize,
                 double *a,
                 double *b,
                 double *c,
                 double scalar,
                 double res);
int verify_ddot(size_t memsize,
                double *a,
                double *b,
                double *c,
                double scalar,
                double res);
int verify_dnrm2(size_t memsize,
                 double *a,
                 double *b,
                 double *c,
                 double scalar,
                 double res);
int verify_dscal(size_t memsize,
                 double *a,
                 double *b,
                 double *c,
                 double scalar,
                 double res);
int verify_dswap(size_t memsize,
                 double *a,
                 double *b,
                 double *c,
                 double scalar,
                 double res);
int verify_drot(size_t memsize,
                double *a,
                double *b,
                double *c,
                double scalar,
                double scalar2,
                double res);
int verify_drotm(size_t memsize,
                 double *a,
                 double *b,
                 double *c,
                 double scalar,
                 double scalar2,
                 double res);
int verify_idmax(size_t memsize,
                 double *a,
                 double *b,
                 double *c,
                 double scalar,
                 double res);
