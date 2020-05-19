/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *******************************************************************************/

#include "utils.h"

static const char *namespace;
static int active = 0;

void log_init(const char *nm)
{
	(void)*nm;
	char *debug = getenv("DEBUG");

	if (debug)
		active = atoi(debug);
}

void log_msg(const char *level, unsigned int line, const char *fmt, ...)
{
	va_list ap;

	if (!active)
		return;
	printf("%s:\t%s:\t%u:\t", namespace, level, line);
	va_start(ap, fmt);
	vprintf(fmt, ap);
	va_end(ap);
}

double mysecond(void)
{
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}
