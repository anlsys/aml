/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *******************************************************************************/

#include <limits.h>

#include <aml.h>

#include "utils.h"

long long int aml_timediff(struct timespec start, struct timespec end)
{
	long long int timediff = (end.tv_nsec - start.tv_nsec) +
	                         1000000000 * (end.tv_sec - start.tv_sec);
	return timediff;
}

int aml_stats_init(struct aml_time_stats *out)
{
	if (out == NULL)
		return -AML_EINVAL;

	out->sum = 0;
	out->max = LONG_MIN;
	out->min = LONG_MAX;
	out->n = 0;

	return AML_SUCCESS;
}

int aml_time_stats_add(struct aml_time_stats *out, const long long val)
{
	if (out == NULL)
		return -AML_EINVAL;

	out->n++;
	out->sum += val;
	out->max = out->max > val ? out->max : val;
	out->min = out->min < val ? out->min : val;

	return AML_SUCCESS;
}
