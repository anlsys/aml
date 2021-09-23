/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#ifndef AML_BENCHS_UTILS_H
#define AML_BENCHS_UTILS_H 1

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_num_threads() 1
#endif

typedef struct timespec aml_time_t;

static inline void aml_gettime(aml_time_t *now)
{
	clock_gettime(CLOCK_REALTIME, now);
}

long long int aml_timediff(aml_time_t start, aml_time_t end);

// Structure holding statistics of a set of time samples.
struct aml_time_stats {
	// The updated mean value.
	long long sum;
	// The maximum time value.
	long long max;
	// The minimum time value.
	long long min;
	// The number of samples in the statistics.
	unsigned long long n;
};

/**
 * Initialize a `struct aml_time_stats` before collecting
 * samples.
 */
int aml_stats_init(struct aml_time_stats *out);

/**
 * Update statistics (online) with an additional sample.
 *
 * @param[in, out] out: The ongoing statistic record on which
 * to compute statistics incrementally with a new sample.
 * @param[in] val: A time value to add to the statistic record.
 */
int aml_time_stats_add(struct aml_time_stats *out, const long long val);

#endif // AML_BENCHS_UTILS_H
