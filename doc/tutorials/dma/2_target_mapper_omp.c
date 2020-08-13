/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#define N 100

typedef struct myvec {
	size_t len;
	double *data;
} myvec_t;

// Requires OpenMP 5.0 or greater. -------------------------------------------//
// #pragma omp declare mapper(myvec_t v) map(v, v.data[0:v.len])
//----------------------------------------------------------------------------//

void init(myvec_t *s)
{
	for (size_t i = 0; i < s->len; i++)
		s->data[i] = i;
}

int main()
{
	myvec_t s;
	s.data = (double *)calloc(N, sizeof(double));
	s.len = N;

#pragma omp target
	init(&s);
	printf("s.data[%d]=%lf\n", N - 1, s.data[N - 1]);
	// s.data[99]=99.000000
}

// OpenMP Examples Version 5.0.0 - November 2019
