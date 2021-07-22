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

#include "aml/dma/linux.h"

struct aml_dma *aml_dma_linux;

int aml_backend_linux_init(void)
{
	return aml_dma_linux_create(&aml_dma_linux, 32);
}

int aml_backend_linux_finalize(void)
{
	aml_dma_linux_destroy(&aml_dma_linux);
	return AML_SUCCESS;
}
