/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include <float.h>

#include "verify_blas_l1.h"

#define MAX(a, b) (((a) < (b)) ? (a) : (b))
#define ABS(a) (((a) < 0) ? -(a) : (a))

#define check_double(ref, value, bits)                                         \
	do {                                                                   \
		double diff = ABS((ref) - (value));                            \
		assert(diff <= MAX(ABS(ref), ABS(value)) * DBL_EPSILON *       \
		                       ((1 << (bits)) - 1));                   \
	} while (0)

void init_arrays(size_t memsize, double *a, double *b, double *c)
{
#pragma omp parallel for
	for (size_t i = 0; i < memsize; i++) {
		a[i] = (double)i;
		b[i] = (double)(memsize - i);
		c[i] = 0.0;
	}
}

int verify_dasum(size_t memsize,
                 double *a,
                 double *b,
                 double *c,
                 double scalar,
                 double res)
{
	(void)*a;
	(void)*b;
	(void)*c;
	(void)scalar;
	check_double((memsize - 1) * memsize / 2.0, res, 2);
	return 1;
}

int verify_daxpy(size_t memsize,
                 double *a,
                 double *b,
                 double *c,
                 double scalar,
                 double res)
{
	(void)*a;
	(void)*b;
	(void)res;
#pragma omp parallel for
	for (size_t i = 0; i < memsize; i++)
		check_double(memsize + (scalar - 1) * i, c[i], 2);
	return 1;
}

int verify_dcopy(size_t memsize,
                 double *a,
                 double *b,
                 double *c,
                 double scalar,
                 double res)
{
	(void)*a;
	(void)*c;
	(void)scalar;
	(void)res;
#pragma omp parallel for
	for (size_t i = 0; i < memsize; i++)
		check_double((double)i, b[i], 2);
	return 1;
}

int verify_ddot(size_t memsize,
                double *a,
                double *b,
                double *c,
                double scalar,
                double res)
{
	(void)*a;
	(void)*b;
	(void)*c;
	(void)scalar;
	check_double(memsize * (memsize * memsize - 1) / 6.0, res, 2);
	return 1;
}

int verify_dnrm2(size_t memsize,
                 double *a,
                 double *b,
                 double *c,
                 double scalar,
                 double res)
{
	(void)*a;
	(void)*b;
	(void)*c;
	(void)scalar;
	check_double(sqrt((memsize - 1) * memsize * (2 * memsize - 1) / 6), res,
	             4);
	return 1;
}

int verify_dscal(size_t memsize,
                 double *a,
                 double *b,
                 double *c,
                 double scalar,
                 double res)
{
	(void)*a;
	(void)*c;
	(void)res;
#pragma omp parallel for
	for (size_t i = 0; i < memsize; i++)
		check_double(scalar * i, b[i], 2);
	return 1;
}

int verify_dswap(size_t memsize,
                 double *a,
                 double *b,
                 double *c,
                 double scalar,
                 double res)
{
	(void)*c;
	(void)scalar;
	(void)res;
#pragma omp parallel for
	for (size_t i = 0; i < memsize; i++) {
		check_double((double)(memsize - i), a[i], 2);
		check_double((double)i, b[i], 2);
	}
	return 1;
}

int verify_drot(size_t memsize,
                double *a,
                double *b,
                double *c,
                double scalar,
                double scalar2,
                double res)
{
	(void)*c;
	(void)res;
#pragma omp parallel for
	for (size_t i = 0; i < memsize; i++) {
		check_double(scalar * i + scalar2 * (memsize - i), a[i], 2);
		check_double(scalar * (memsize - i) - scalar2 * i, b[i], 2);
	}
	return 1;
}

int verify_drotm(size_t memsize,
                 double *a,
                 double *b,
                 double *c,
                 double scalar,
                 double scalar2,
                 double res)
{
	(void)*c;
	(void)res;
	(void)scalar;
	(void)scalar2;
#pragma omp parallel for
	for (size_t i = 0; i < memsize; i++) {
		check_double((double)(i + 3 * (memsize - i)), a[i], 2);
		check_double((double)(2 * i + 4 * (memsize - i)), b[i], 2);
	}
	return 1;
}

int verify_idmax(size_t memsize,
                 double *a,
                 double *b,
                 double *c,
                 double scalar,
                 double res)
{
	(void)*a;
	(void)*b;
	(void)*c;
	(void)scalar;
	check_double((double)(memsize - 1), res, 2);
	return 1;
}
