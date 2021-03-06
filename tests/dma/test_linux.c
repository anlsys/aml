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

#include "aml/area/linux.h"
#include "aml/dma/linux.h"

int main(int argc, char **argv)
{
	assert(aml_init(&argc, &argv) == AML_SUCCESS);

	test_dma_memcpy(&aml_area_linux, NULL, aml_dma_linux,
	                aml_dma_linux_memcpy_op);

	test_dma_barrier(&aml_area_linux, NULL, aml_dma_linux,
	                 aml_dma_linux_memcpy_op);

	aml_finalize();
}
