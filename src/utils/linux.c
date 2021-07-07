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

#include "aml/dma/linux-par.h"
#include "aml/dma/linux-seq.h"

struct aml_dma *aml_dma_linux_sequential;
struct aml_dma *aml_dma_linux_parallel;

int aml_linux_init(void)
{
	int err;

	err = aml_dma_linux_seq_create(&aml_dma_linux_sequential, 64, NULL,
	                               NULL);
	if (err != AML_SUCCESS)
		return err;

	err = aml_dma_linux_par_create(&aml_dma_linux_parallel, 64, NULL, NULL);
	if (err != AML_SUCCESS) {
		aml_dma_linux_seq_destroy(&aml_dma_linux_sequential);
		return err;
	}

	return AML_SUCCESS;
}

int aml_linux_finalize(void)
{
	aml_dma_linux_seq_destroy(&aml_dma_linux_sequential);
	aml_dma_linux_par_destroy(&aml_dma_linux_parallel);
	return AML_SUCCESS;
}
