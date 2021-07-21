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
#include "aml/dma/linux.h"

struct aml_dma *aml_dma_linux_sequential;
struct aml_dma *aml_dma_linux_parallel;
struct aml_dma *aml_dma_linux;

int aml_backend_linux_init(void)
{
	int err;

	err = aml_dma_linux_seq_create(&aml_dma_linux_sequential, 64, NULL,
	                               NULL);
	if (err != AML_SUCCESS)
		return err;

	err = aml_dma_linux_par_create(&aml_dma_linux_parallel, 64, NULL, NULL);
	if (err != AML_SUCCESS)
		goto err_with_dma_seq;

	err = aml_dma_linux_create(&aml_dma_linux, 32);
	if (err != AML_SUCCESS)
		goto err_with_dma_par;

	return AML_SUCCESS;

err_with_dma_par:
	aml_dma_linux_seq_destroy(&aml_dma_linux_parallel);
err_with_dma_seq:
	aml_dma_linux_seq_destroy(&aml_dma_linux_sequential);
	return err;
}

int aml_backend_linux_finalize(void)
{
	aml_dma_linux_seq_destroy(&aml_dma_linux_sequential);
	aml_dma_linux_par_destroy(&aml_dma_linux_parallel);
	aml_dma_linux_destroy(&aml_dma_linux);
	return AML_SUCCESS;
}
