/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
*******************************************************************************/

#include "aml.h"
#include <stdio.h>

static const char *const aml_error_strings[] = {
        [AML_SUCCESS] = "Success",
        [AML_FAILURE] = "Generic error",
        [AML_ENOMEM] = "Not enough memory",
        [AML_EINVAL] = "Invalid argument",
        [AML_EDOM] = "Value out of bound",
        [AML_EBUSY] = "Underlying resource is not available for operation",
        [AML_ENOTSUP] = "Operation not supported",
        [AML_EPERM] = "Insufficient permissions",
};

const char *aml_strerror(const int err)
{
	if (err < 0 || err >= AML_ERROR_MAX)
		return "Unknown error";
	return aml_error_strings[err];
}

void aml_perror(const char *msg)
{
	fprintf(stderr, "%s:%s\n", msg, aml_strerror(aml_errno));
}
