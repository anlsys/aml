/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
*******************************************************************************/

#include "aml.h"
#include <assert.h>
#include <errno.h>
#include <mkl.h>
#include <omp.h>
#include <pthread.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>


int main(int argc, char *argv[])
{
	struct aml_bitmap slow_b, fast_b;
	aml_bitmap_zero(&slow_b);
	aml_bitmap_zero(&fast_b);
	aml_bitmap_set(&slow_b, 0);
	aml_bitmap_set(&fast_b, 1);
	struct aml_area * slow = aml_local_area_create(aml_area_linux_private, &slow_b, 0);
	struct aml_area * fast = aml_local_area_create(aml_area_linux_private, &fast_b, 0);

	struct timespec start, stop;
	double *a, *b, *c;
	aml_init(&argc, &argv);
	assert(argc == 4);
	long int N = atol(argv[3]);
	unsigned long memsize = sizeof(double)*N*N;

	assert(aml_area_malloc(slow, (void**)(&a), memsize, 0) == AML_AREA_SUCCESS);
	assert(aml_area_malloc(slow, (void**)(&b), memsize, 0) == AML_AREA_SUCCESS);
	assert(aml_area_malloc(fast, (void**)(&c), memsize, 0) == AML_AREA_SUCCESS);
	assert(a != NULL && b != NULL && c != NULL);

	double alpha = 1.0, beta = 1.0;
	for(unsigned long i = 0; i < N*N; i++){
		a[i] = (double)rand();
		b[i] = (double)rand();
		c[i] = 0.0;
	}

	clock_gettime(CLOCK_REALTIME, &start);
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, alpha, a, N, b, N, beta, c, N);
	clock_gettime(CLOCK_REALTIME, &stop);
	long long int time = 0;
	time =  (stop.tv_nsec - start.tv_nsec) +
                1e9* (stop.tv_sec - start.tv_sec);
	double flops = (2.0*N*N*N)/(time/1e9);
	/* print the flops in GFLOPS */
	printf("dgemm-mkl: %llu %lld %lld %f\n", N, memsize, time, flops/1e9);
	aml_area_free(slow, a);
	aml_area_free(slow, b);
	aml_area_free(fast, c);
	aml_local_area_destroy(slow);
	aml_local_area_destroy(fast);
	aml_finalize();
	return 0;
}
