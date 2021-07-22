/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include "aml.h"

#include "aml/area/ze.h"
#include "aml/dma/ze.h"
#include "aml/utils/backend/ze.h"
#define ZE(ze_call) aml_errno_from_ze_result(ze_call)

#include "test_dma.h"

int main(int argc, char **argv)
{
	assert(aml_init(&argc, &argv) == AML_SUCCESS);
	if (!aml_support_backends(AML_BACKEND_ZE))
		return 77;

	test_dma_memcpy(aml_area_ze_device, NULL, aml_dma_ze_default,
	                aml_memcpy_ze);

	test_dma_barrier(aml_area_ze_device, NULL, aml_dma_ze_default,
	                 aml_memcpy_ze);

	aml_finalize();
}
