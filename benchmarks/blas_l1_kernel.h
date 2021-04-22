/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/
/*
 * This is a benchmark for the BLAS Level 1 operations for AML.
 */

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <unistd.h>

double dasum(size_t n, double *a, double *b, double *c, double scalar);
double daxpy(size_t n, double *a, double *b, double *c, double scalar);
double dcopy(size_t n, double *a, double *b, double *c, double scalar);
double ddot(size_t n, double *a, double *b, double *c, double scalar);
double dnrm2(size_t n, double *a, double *b, double *c, double scalar);
double dscal(size_t n, double *a, double *b, double *c, double scalar);
double dswap(size_t n, double *a, double *b, double *c, double scalar);
double idmax(size_t n, double *a, double *b, double *c, double scalar);
void drot(size_t n, double *a, double *b, double c, double s);
void drotg(double x, double y, double c, double s);
void drotm(size_t n, double *a, double *b, double *param);
void drotmg(double d1, double d2, double x, double y, double *param);
