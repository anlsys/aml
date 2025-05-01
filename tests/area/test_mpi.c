/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *******************************************************************************/
#include <stdlib.h>
#include <string.h>

#include "aml.h"
#include "aml/area/mpi.h"

int main(void)
{
	assert(aml_init(NULL, NULL) == AML_SUCCESS);
	if (!aml_support_backends(AML_BACKEND_MPI))
		return 77;
	aml_finalize();
	return 0;
}
