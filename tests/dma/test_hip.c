/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include "test_dma.h"

#include "aml.h"

#include "aml/area/hip.h"
#include "aml/dma/hip.h"

int main(int argc, char **argv)
{
	assert(aml_init(&argc, &argv) == AML_SUCCESS);
	if (!aml_support_backends(AML_BACKEND_HIP))
		return 77;

	test_dma_memcpy(&aml_area_hip, NULL, &aml_dma_hip,
	                aml_dma_hip_memcpy_op);

	test_dma_barrier(&aml_area_hip, NULL, &aml_dma_hip,
	                 aml_dma_hip_memcpy_op);

	aml_finalize();
}
