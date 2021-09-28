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
#include <errno.h>
#include <stdio.h>
#include <unistd.h>

#include "aml.h"

#include "aml/area/linux.h"
#include "aml/layout/dense.h"
#include "aml/tiling/resize.h"

#include "blas/l1_kernel.h"
#include "blas/verify_l1.h"
#include "utils.h"

#define DEFAULT_ARRAY_SIZE (1UL << 20)
#define DEFAULT_TILE_SIZE (1UL << 8)

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

#ifndef abs
#define abs(a) ((a) >= 0 ? (a) : -(a))
#endif

static double *pt;

double run_dasum(size_t tilesize,
                 size_t ntiles,
                 struct aml_tiling *ta,
                 struct aml_tiling *tb,
                 struct aml_tiling *tc,
                 double scalar)
{
	(void)*tb;
	(void)*tc;
	double asum = 0;

#pragma omp parallel for reduction(+ : asum)
	for (size_t i = 0; i < ntiles; i++) {
		double *at, *bt, *ct;
		double temp;

		at = aml_tiling_rawptr(ta, (size_t[]){i});
		bt = pt;
		ct = pt;

		temp = dasum(tilesize, at, bt, ct, scalar);
		asum += temp;
	}
	return asum;
}

double run_daxpy(size_t tilesize,
                 size_t ntiles,
                 struct aml_tiling *ta,
                 struct aml_tiling *tb,
                 struct aml_tiling *tc,
                 double scalar)
{
#pragma omp parallel for
	for (size_t i = 0; i < ntiles; i++) {
		double *at, *bt, *ct;

		at = aml_tiling_rawptr(ta, (size_t[]){i});
		bt = aml_tiling_rawptr(tb, (size_t[]){i});
		ct = aml_tiling_rawptr(tc, (size_t[]){i});

		daxpy(tilesize, at, bt, ct, scalar);
	}
	return 1;
}

double run_dcopy(size_t tilesize,
                 size_t ntiles,
                 struct aml_tiling *ta,
                 struct aml_tiling *tb,
                 struct aml_tiling *tc,
                 double scalar)
{
	(void)*tc;
#pragma omp parallel for
	for (size_t i = 0; i < ntiles; i++) {
		double *at, *bt, *ct;

		at = aml_tiling_rawptr(ta, (size_t[]){i});
		bt = aml_tiling_rawptr(tb, (size_t[]){i});
		ct = pt;

		dcopy(tilesize, at, bt, ct, scalar);
	}
	return 1;
}

double run_ddot(size_t tilesize,
                size_t ntiles,
                struct aml_tiling *ta,
                struct aml_tiling *tb,
                struct aml_tiling *tc,
                double scalar)
{
	(void)*tc;
	double dot = 0.0;

#pragma omp parallel for reduction(+ : dot)
	for (size_t i = 0; i < ntiles; i++) {
		double *at, *bt, *ct;
		double temp;

		at = aml_tiling_rawptr(ta, (size_t[]){i});
		bt = aml_tiling_rawptr(tb, (size_t[]){i});
		ct = pt;

		temp = ddot(tilesize, at, bt, ct, scalar);
		dot += temp;
	}
	return dot;
}

double run_dnrm2(size_t tilesize,
                 size_t ntiles,
                 struct aml_tiling *ta,
                 struct aml_tiling *tb,
                 struct aml_tiling *tc,
                 double scalar)
{
	(void)*tb;
	(void)*tc;
	double nrm2 = 0;

#pragma omp parallel for reduction(+ : nrm2)
	for (size_t i = 0; i < ntiles; i++) {
		double *at, *bt, *ct;
		double nrm;

		at = aml_tiling_rawptr(ta, (size_t[]){i});
		bt = pt;
		ct = pt;

		nrm = dnrm2(tilesize, at, bt, ct, scalar);
		nrm2 += pow(nrm, 2);
	}
	return sqrt(nrm2);
}

double run_dscal(size_t tilesize,
                 size_t ntiles,
                 struct aml_tiling *ta,
                 struct aml_tiling *tb,
                 struct aml_tiling *tc,
                 double scalar)
{
	(void)*tc;
#pragma omp parallel for
	for (size_t i = 0; i < ntiles; i++) {
		double *at, *bt, *ct;

		at = aml_tiling_rawptr(ta, (size_t[]){i});
		bt = aml_tiling_rawptr(tb, (size_t[]){i});
		ct = pt;

		dscal(tilesize, at, bt, ct, scalar);
	}
	return 1;
}

double run_dswap(size_t tilesize,
                 size_t ntiles,
                 struct aml_tiling *ta,
                 struct aml_tiling *tb,
                 struct aml_tiling *tc,
                 double scalar)
{
	(void)*tc;
#pragma omp parallel for
	for (size_t i = 0; i < ntiles; i++) {
		double *at, *bt, *ct;

		at = aml_tiling_rawptr(ta, (size_t[]){i});
		bt = aml_tiling_rawptr(tb, (size_t[]){i});
		ct = pt;

		dswap(tilesize, at, bt, ct, scalar);
	}
	return 1;
}

double run_idmax(size_t tilesize,
                 size_t ntiles,
                 struct aml_tiling *ta,
                 struct aml_tiling *tb,
                 struct aml_tiling *tc,
                 double scalar)
{
	(void)*tb;
	(void)*tc;
	size_t maxid = 0;
	double max = 0.0;

#pragma omp parallel
	{
		double local_max = -DBL_MAX;
		size_t local_maxid;
#pragma omp parallel for
		for (size_t i = 0; i < ntiles; i++) {
			double *at, *bt, *ct;
			double maxl;
			size_t maxidl;

			at = aml_tiling_rawptr(ta, (size_t[]){i});
			bt = pt;
			ct = pt;

			maxidl = idmax(tilesize, at, bt, ct, scalar);
			maxl = abs(at[maxidl]);
			maxidl += i * tilesize;

			if (local_max < maxl) {
				local_max = maxl;
				local_maxid = maxidl;
			}
		}
#pragma omp critical
		if (max < local_max) {
			max = local_max;
			maxid = local_maxid;
		}
	}
	return maxid;
}

double run_drot(size_t tilesize,
                size_t ntiles,
                struct aml_tiling *ta,
                struct aml_tiling *tb,
                struct aml_tiling *tc,
                double x,
                double y)
{
	(void)*tc;
#pragma omp parallel for
	for (size_t i = 0; i < ntiles; i++) {
		double *at, *bt;

		at = aml_tiling_rawptr(ta, (size_t[]){i});
		bt = aml_tiling_rawptr(tb, (size_t[]){i});

		drot(tilesize, at, bt, x, y);
	}
	return 1;
}

// TODO implement drotg(x, y, w, s);

double run_drotm(size_t tilesize,
                 size_t ntiles,
                 struct aml_tiling *ta,
                 struct aml_tiling *tb,
                 struct aml_tiling *tc,
                 double *param)
{
	(void)*tc;
#pragma omp parallel for
	for (size_t i = 0; i < ntiles; i++) {
		double *at, *bt;

		at = aml_tiling_rawptr(ta, (size_t[]){i});
		bt = aml_tiling_rawptr(tb, (size_t[]){i});

		drotm(tilesize, at, bt, param);
	}
	return 1;
}

// TODO implement drotmg(d1, d2, x, y, param);

typedef double (*r)(size_t,
                    size_t,
                    struct aml_tiling *,
                    struct aml_tiling *,
                    struct aml_tiling *,
                    double);

r run_f[8] = {&run_dcopy, &run_dscal, &run_daxpy, &run_dasum,
              &run_ddot,  &run_dnrm2, &run_dswap, &run_idmax};
v verify_f[8] = {&verify_dcopy, &verify_dscal, &verify_daxpy, &verify_dasum,
                 &verify_ddot,  &verify_dnrm2, &verify_dswap, &verify_idmax};

int main(int argc, char *argv[])
{
	aml_init(&argc, &argv);
	struct aml_area *area = &aml_area_linux;
	size_t nb_reps;
	size_t memsize, tilesize, ntiles;
	size_t i, j, k;
	long long int timing;
	aml_time_t start, end;
	double *a, *b, *c;
	struct aml_layout *la, *lb, *lc;
	struct aml_tiling *ta, *tb, *tc;
	double res;
	double scalar = 1.0;
	double scal2 = 2.0;
	double param[5];

	param[0] = -1.0;
	for (size_t i = 1; i < 5; i++)
		param[i] = i;

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
		tilesize = DEFAULT_TILE_SIZE;
		nb_reps = NTIMES;
	} else {
		assert(argc == 3);
		memsize = 1UL << atoi(argv[1]);
		tilesize = 1UL << atoi(argv[2]);
		nb_reps = atoi(argv[3]);
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

	// AML code
	a = aml_area_mmap(area, memsize * sizeof(double), NULL);
	b = aml_area_mmap(area, memsize * sizeof(double), NULL);
	c = aml_area_mmap(area, memsize * sizeof(double), NULL);
	assert(a != NULL && b != NULL && c != NULL);

	/* layouts */
	assert(!aml_layout_dense_create(&la, a, AML_LAYOUT_ORDER_C,
	                                sizeof(double), 1, (size_t[]){memsize},
	                                NULL, NULL));
	assert(!aml_layout_dense_create(&lb, b, AML_LAYOUT_ORDER_C,
	                                sizeof(double), 1, (size_t[]){memsize},
	                                NULL, NULL));
	assert(!aml_layout_dense_create(&lc, c, AML_LAYOUT_ORDER_C,
	                                sizeof(double), 1, (size_t[]){memsize},
	                                NULL, NULL));
	assert(la != NULL && lb != NULL && lc != NULL);
	/* tilings */
	assert(!aml_tiling_resize_create(&ta, AML_TILING_ORDER_C, la, 1,
	                                 (size_t[]){tilesize}));
	assert(!aml_tiling_resize_create(&tb, AML_TILING_ORDER_C, lb, 1,
	                                 (size_t[]){tilesize}));
	assert(!aml_tiling_resize_create(&tc, AML_TILING_ORDER_C, lc, 1,
	                                 (size_t[]){tilesize}));
	assert(ta != NULL && tb != NULL && tc != NULL);
	aml_tiling_dims(ta, &ntiles);

	/* MAIN LOOP - repeat test cases nb_reps */
	for (k = 0; k < nb_reps; k++) {
		// Trying this array of functions thing
		for (i = 0; i < 8; i++) {
			init_arrays(memsize, a, b, c);
			aml_gettime(&start);
			res = run_f[i](tilesize, ntiles, ta, tb, tc, scalar);
			aml_gettime(&end);
			timing = aml_timediff(start, end);
			verify_f[i](memsize, a, b, c, scalar, res);
			sumtime[i] += timing;
			mintime[i] = MIN(mintime[i], timing);
			maxtime[i] = MAX(maxtime[i], timing);
		}

		// Rotations
		init_arrays(memsize, a, b, c);
		aml_gettime(&start);
		res = run_drot(tilesize, ntiles, ta, tb, tc, scal2, scalar);
		aml_gettime(&end);
		timing = aml_timediff(start, end);
		verify_drot(memsize, a, b, c, scal2, scalar, res);
		sumtime[8] += timing;
		mintime[8] = MIN(mintime[i], timing);
		maxtime[8] = MAX(maxtime[i], timing);

		init_arrays(memsize, a, b, c);
		aml_gettime(&start);
		res = run_drotm(tilesize, ntiles, ta, tb, tc, param);
		aml_gettime(&end);
		timing = aml_timediff(start, end);
		verify_drotm(memsize, a, b, c, scal2, scalar, res);
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

	/* destroy everything */
	aml_tiling_resize_destroy(&ta);
	aml_tiling_resize_destroy(&tb);
	aml_tiling_resize_destroy(&tc);
	aml_layout_destroy(&la);
	aml_layout_destroy(&lb);
	aml_layout_destroy(&lc);
	aml_area_munmap(area, a, memsize * sizeof(double));
	aml_area_munmap(area, b, memsize * sizeof(double));
	aml_area_munmap(area, c, memsize * sizeof(double));
	aml_finalize();
	return 0;
}
