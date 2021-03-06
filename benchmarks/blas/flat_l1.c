/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *******************************************************************************/
/*
 * This is a benchmark for the BLAS Level 1 operations for AML.
 */

#include <assert.h>
#include <stdio.h>
#include <unistd.h>

#include "aml.h"

#include "aml/area/linux.h"

#include "blas/l1_kernel.h"
#include "blas/verify_l1.h"
#include "utils.h"

/* Look into another way to define these parameters */

#define DEFAULT_ARRAY_SIZE (1UL << 15)

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

static double *a, *b, *c;

typedef double (*r)(size_t, double *, double *, double *, double);

r run_f[8] = {&dcopy, &dscal, &daxpy, &dasum, &ddot, &dnrm2, &dswap, &idmax};
v verify_f[8] = {&verify_dcopy, &verify_dscal, &verify_daxpy, &verify_dasum,
                 &verify_ddot,  &verify_dnrm2, &verify_dswap, &verify_idmax};

int main(int argc, char *argv[])
{
	aml_init(&argc, &argv);
	size_t i, j, k;
	size_t nb_reps;
	size_t memsize;
	double dscalar;
	long long int timing;
	long long int sumtime[10] = {0}, maxtime[10] = {0},
	              mintime[10] = {LONG_MAX, LONG_MAX, LONG_MAX, LONG_MAX,
	                             LONG_MAX, LONG_MAX, LONG_MAX, LONG_MAX,
	                             LONG_MAX, LONG_MAX};
	char *label[10] = {
	        "Copy:	", "Scale:	", "Triad:	", "Asum:	",
	        "Dot:	", "Norm:	", "Swap:	", "Max ID:	",
	        "RotP:	", "RotM:	"};

	if (argc == 1) {
		memsize = DEFAULT_ARRAY_SIZE;
		nb_reps = NTIMES;
	} else {
		assert(argc == 2);
		memsize = 1UL << atoi(argv[1]);
		nb_reps = atoi(argv[2]);
	}

	printf("Each kernel will be executed %ld times.\n", nb_reps);

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
	// AML code
	struct aml_area *area = &aml_area_linux;

	a = aml_area_mmap(area, size, NULL);
	b = aml_area_mmap(area, size, NULL);
	c = aml_area_mmap(area, size, NULL);

	/* MAIN LOOP - repeat test cases nb_reps */
	dscalar = 3.0;
	double x = 1.0, y = 2.0;
	double param[5];

	param[0] = -1.0;
	for (k = 1; k < 5; k++)
		param[k] = k;
	double res;

	aml_time_t start, end;

	for (k = 0; k < nb_reps; k++) {
		// Trying this array of functions thing
		for (i = 0; i < 8; i++) {
			init_arrays(memsize, a, b, c);
			aml_gettime(&start);
			res = run_f[i](memsize, a, b, c, dscalar);
			aml_gettime(&end);
			timing = aml_timediff(start, end);
			verify_f[i](memsize, a, b, c, dscalar, res);
			sumtime[i] += timing;
			mintime[i] = MIN(mintime[i], timing);
			maxtime[i] = MAX(maxtime[i], timing);
		}

		// Rotations
		init_arrays(memsize, a, b, c);
		aml_gettime(&start);
		drot(memsize, a, b, x, y);
		aml_gettime(&end);
		timing = aml_timediff(start, end);
		verify_drot(memsize, a, b, c, x, y, res);
		sumtime[8] += timing;
		mintime[8] = MIN(mintime[i], timing);
		maxtime[8] = MAX(maxtime[i], timing);

		init_arrays(memsize, a, b, c);
		aml_gettime(&start);
		drotm(memsize, a, b, param);
		aml_gettime(&end);
		timing = aml_timediff(start, end);
		verify_drotm(memsize, a, b, c, x, y, res);
		sumtime[9] += timing;
		mintime[9] = MIN(mintime[i], timing);
		maxtime[9] = MAX(maxtime[i], timing);

		/* Add the rotation generations later, + 2 functions
		drotg(x, y, dc, ds);
		drotmg(d1, d2, x, y, param);
		*/
	}

	/* SUMMARY */
	printf("Function	Avg time	Min time	Max time\n");
	for (j = 0; j < 10; j++) {
		double avg = (double)sumtime[j] / (double)(nb_reps - 1);
		printf("%s\t%11.6f\t%lld\t%lld\n", label[j], avg, mintime[j],
		       maxtime[j]);
	}

	/* aml specific code */
	aml_area_munmap(area, a, size);
	aml_area_munmap(area, b, size);
	aml_area_munmap(area, c, size);
	aml_finalize();

	return 0;
}
