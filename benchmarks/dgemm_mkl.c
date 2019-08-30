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
#include "aml/area/linux.h"
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
	struct aml_area *slow, *fast;
	struct aml_bitmap slowb, fastb;
	struct timespec start, stop;
	double *a, *b, *c;
	aml_init(&argc, &argv);
	assert(argc == 4);
	assert(aml_bitmap_from_string(&fastb, argv[1]) == 0);
	assert(aml_bitmap_from_string(&slowb, argv[2]) == 0);
	long int N = atol(argv[3]);
	unsigned long memsize = sizeof(double)*N*N;

	aml_area_linux_create(&slow, &slowb, AML_AREA_LINUX_POLICY_BIND);
	assert(slow != NULL);
	aml_area_linux_create(&fast, &fastb, AML_AREA_LINUX_POLICY_BIND);
	assert(fast != NULL);

	a = aml_area_mmap(slow, memsize, NULL);
	b = aml_area_mmap(slow, memsize, NULL);
	c = aml_area_mmap(fast, memsize, NULL);
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
	aml_area_munmap(slow, a, memsize);
	aml_area_munmap(slow, b, memsize);
	aml_area_munmap(fast, c, memsize);
	aml_area_linux_destroy(&slow);
	aml_area_linux_destroy(&fast);
	aml_finalize();
	return 0;
}
