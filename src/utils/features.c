/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *******************************************************************************/

#include "config.h"

#include "aml/utils/features.h"
#if HAVE_CUDA == 1
#include <cuda.h>
#include <cuda_runtime.h>
#endif
#if HAVE_HWLOC == 1
#include <hwloc.h>
extern hwloc_topology_t aml_topology;
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

static int aml_support_hwloc(void)
{
#if HAVE_HWLOC == 0
	return 0;
#else
	const struct hwloc_topology_support *sup =
	        hwloc_topology_get_support(aml_topology);
	if (!sup->discovery->numa || !sup->discovery->numa_memory)
		return 0;
	if (!sup->membind->set_area_membind || !sup->membind->alloc_membind)
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

	// hwloc check: compilation support and runtime support must be present.
	if ((backends & AML_BACKEND_HWLOC) &&
	    !(AML_HAVE_BACKEND_HWLOC && aml_support_hwloc()))
		return 0;

	return 1;
}
