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
#include <string.h>

const char* aml_version_string = VERSION;
int         aml_major_version = -1;
int         aml_minor_version = -1;
int         aml_patch_version = -1;

int aml_init(int *argc, char **argv[])
{
	char * version = VERSION;
	aml_major_version = atoi(strtok(version, "."));
	aml_minor_version = atoi(strtok(NULL, "."));
	aml_patch_version = atoi(strtok(NULL, "."));
	return 0;
}

int aml_finalize(void)
{
	return 0;
}

