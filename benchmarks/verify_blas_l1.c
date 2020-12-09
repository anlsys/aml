/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include "verify_blas_l1.h"

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
	assert(res == (memsize - 1) * memsize / 2.0);
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
		assert(c[i] == memsize + (scalar - 1) * i);
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
		assert(b[i] == (double)i);
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
	assert(res == memsize * (memsize * memsize - 1) / 6.0);
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
	assert(res - sqrt((memsize - 1) * memsize * (2 * memsize - 1) / 6) <=
	       1);
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
		assert(b[i] == scalar * i);
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
		assert(a[i] == (double)(memsize - i));
		assert(b[i] == (double)i);
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
		assert(a[i] == (scalar * i + scalar2 * (memsize - i)));
		assert(b[i] == (scalar * (memsize - i) - scalar2 * i));
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
		assert(a[i] == i + 3 * (memsize - i));
		assert(b[i] == 2 * i + 4 * (memsize - i));
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
	assert(res == memsize - 1);
	return 1;
}
