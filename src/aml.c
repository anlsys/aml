/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
*******************************************************************************/

#include "config.h"
#include "aml.h"

size_t aml_pagesize = 4096;

int aml_init(int *argc, char **argv[])
{
	aml_pagesize = sysconf(_SC_PAGE_SIZE);
	return 0;	
}

int aml_finalize(void)
{
	return 0;
}

