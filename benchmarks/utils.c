/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *******************************************************************************/

#include "utils.h"

long long int aml_timediff(struct timespec start, struct timespec end)
{
	long long int timediff = (end.tv_nsec - start.tv_nsec) +
	                         1000000000 * (end.tv_sec - start.tv_sec);
	return timediff;
}
