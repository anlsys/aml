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

long long int aml_timediff(struct timespec start, struct timespec end);

#endif // AML_BENCHS_UTILS_H
