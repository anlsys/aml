/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
*******************************************************************************/

#include "aml/utils/features.h"
#include "config.h"
#if HAVE_CUDA == 1
#include <cuda.h>
#include <cuda_runtime.h>
#endif

static int aml_support_cuda(void)
{
#if HAVE_CUDA == 0
	return 0;
#else
	int x;

	if (cudaGetDeviceCount(&x) != cudaSuccess || x <= 0)
		return 0;

	return 1;
#endif
}

int aml_support_backends(const unsigned long backends)
{
	// Cuda check: compilation support and runtime support must be present.
	if ((backends & AML_BACKEND_CUDA) &&
	    !(AML_HAVE_BACKEND_CUDA && aml_support_cuda()))
		return 0;

	return 1;
}
