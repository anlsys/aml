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

#include <assert.h>
#include <stdio.h>
#include <unistd.h>

#include "aml.h"

#include "aml/area/linux.h"

#include "blas_l2_kernel.h"
#include "utils.h"
#include "verify_blas_l2.h"

/* Look into another way to define these parameters */

#define DEFAULT_ARRAY_SIZE (1UL << 10)

#ifdef NTIMES
#if NTIMES <= 1
#define NTIMES 10
#endif
#endif
#ifndef NTIMES
#define NTIMES 10
#endif
#define OFFSET 0

#ifndef MIN
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#endif
#ifndef MAX
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#endif

extern int omp_get_num_threads(void);

typedef double (*r)(bool,
                    bool,
                    bool,
                    size_t,
                    size_t,
                    int,
                    int,
                    double,
                    double,
                    double **,
                    double *,
                    double *,
                    double *);

r run_f[15] = {&dgbmv, &dgemv, &dger,  &dsbmv, &dspmv, &dspr,  &dspr2, &dsymv,
               &dsyr,  &dsyr2, &dtbmv, &dtbsv, &dtpmv, &dtpsv, &dtrmv};

// TODO implement those functions
v verify_f[15] = {&verify_dgbmv, &verify_dgemv, &verify_dger,  &verify_dsbmv,
                  &verify_dspmv, &verify_dspr,  &verify_dspr2, &verify_dsymv,
                  &verify_dsyr,  &verify_dsyr2, &verify_dtbmv, &verify_dtbsv,
                  &verify_dtpmv, &verify_dtpsv, &verify_dtrmv};

int main(int argc, char *argv[])
{
	aml_init(&argc, &argv);
	struct aml_area *area = &aml_area_linux;
	size_t i, j, k;
	size_t memsize;
	double timing; //, t;
	double avgtime[15] = {0}, maxtime[15] = {0},
	       mintime[15] = {FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX,
	                      FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX,
	                      FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX};

	char *label[15] = {
	        "Dgbmv:	", "Dgemv:	", "Dger:	", "Dsbmv:	",
	        "Dspmv:	", "Dspr:	", "Dspr2:	", "Dsymv:	",
	        "Dsyr:	", "Dsyr2:	", "Dtbmv:	", "Dtbsv:	",
	        "Dtpmv,	", "Dtpsv:	", "Dtrmv:	"};

	if (argc == 1) {
		memsize = DEFAULT_ARRAY_SIZE;
	} else {
		assert(argc == 2);
		memsize = 1UL << atoi(argv[1]);
	}

	bool trans, uplo, unit;
	size_t m, n;
	int kl, ku;
	double alpha, beta;
	double *x, *y, *at;
	double **a;

#pragma omp parallel
	{
#pragma omp master
		{
			k = omp_get_num_threads();
			printf("Number of threads required = %li\n", k);
		}
	}

	k = 0;
#pragma omp parallel
#pragma omp atomic
	k++;
	printf("Number of threads counted = %li\n", k);

	size_t size = sizeof(double) * (memsize + OFFSET);
	a = aml_area_mmap(area, size ^ 2, NULL);
	x = aml_area_mmap(area, size, NULL);
	y = aml_area_mmap(area, size, NULL);
	at = aml_area_mmap(area, size ^ 2, NULL);

	/* MAIN LOOP - repeat test cases NTIMES */
	m = memsize - 2;
	n = memsize;
	alpha = 2.0;
	beta = 3.0;
	trans = 0;
	uplo = 1;
	unit = 0;
	kl = 2;
	ku = 2;

	for (k = 0; k < NTIMES; k++) {
		// Array of functions
		for (i = 0; i < 15; i++) {
			// TODO Implement both init functions
			if (i < 3)
				init_matrix_mn(m, n, kl, ku, a, x, y, at);
			else
				init_matrix_n(n, kl, ku, a, x, y, at);

			timing = mysecond();

			res = run_f[i](trans, uplo, unit, m, n, kl, ku, alpha,
			               beta, a, x, y, at);

			timing = mysecond() - timing;

			// TODO Change the arguments once implemented
			verify_f[i](m, n, kl, ku, alpha, beta, a, x, y, at);

			avgtime[i] += timing;
			mintime[i] = MIN(mintime[i], timing);
			maxtime[i] = MAX(maxtime[i], timing);
		}
	}

	/* SUMMARY */
	printf("Function	Avg time	Min time	Max time\n");
	for (j = 0; j < 15; j++) {
		avgtime[j] = avgtime[j] / (double)(NTIMES - 1);
		printf("%s	%11.6f	%11.6f	%11.6f\n", label[j], avgtime[j],
		       mintime[j], maxtime[j]);
	}

	/* aml specific code */
	aml_area_munmap(area, a, size ^ 2);
	aml_area_munmap(area, x, size);
	aml_area_munmap(area, y, size);
	aml_area_munmap(area, at, size ^ 2);
	aml_finalize();

	return 0;
}
