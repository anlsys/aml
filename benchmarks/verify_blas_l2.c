/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include "verify_blas_l2.h"

void init_arrays(size_t memsize, double *x, double *y, double *z)
{
        #pragma omp parallel for
        for (size_t i = 0; i < memsize; i++) {
        	x[i] = (double)i;
        	y[i] = (double)(memsize - i);
        	z[i] = 0.0;
        }
}

void init_matrix(size_t m, size_t n,  double **a)
{
        #pragma omp parallel for
        for (size_t i = 0; i < m; i++) {
		for (size_t j = 0; j < n; j++) {
			a[i][j] = (double)(i+j);
		}
        }
}


