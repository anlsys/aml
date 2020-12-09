/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

#ifndef AML_BENCHS_UTILS_H
#define AML_BENCHS_UTILS_H 1

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_num_threads() 1
#endif

void log_init(const char *nm);
void log_msg(const char *level, unsigned int line, const char *fmt, ...);
double mysecond(void);

#define debug(...) log_msg("debug", __LINE__, __VA_ARGS__)

#endif // AML_BENCHS_UTILS_H
