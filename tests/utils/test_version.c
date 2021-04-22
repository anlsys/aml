/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
*******************************************************************************/

#include "aml.h"
#include <assert.h>

int main(void)
{
	assert(aml_version_major == AML_VERSION_MAJOR);
	assert(aml_version_minor == AML_VERSION_MINOR);
	assert(aml_version_patch == AML_VERSION_PATCH);
	assert(!strcmp(aml_version_string, AML_VERSION_STRING));
	return 0;
}

