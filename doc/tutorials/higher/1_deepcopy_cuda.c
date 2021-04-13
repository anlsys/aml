/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *******************************************************************************/

//- Keep previous tutorial ----------------------------------------------------

#define DEEP_COPY_CUDA
#include "0_deepcopy_linux.c"

//- Cuda specific allocation / data movement ----------------------------------

#include "aml/area/cuda.h"
#include "aml/dma/cuda.h"

//- Mapping args ---------------------------------------------------------------

struct aml_mapper_args cuda_host_to_device_mapper_args;
struct aml_mapper_args cuda_device_to_host_mapper_args;

int main(int argc, char **argv)
{
	struct C *c, *_c, *__c;

	assert(aml_init(&argc, &argv) == AML_SUCCESS);
	if (!aml_support_backends(AML_BACKEND_CUDA))
		return 77;

	// Mapping args initialization.
	cuda_host_to_device_mapper_args.area = &aml_area_cuda;
	cuda_host_to_device_mapper_args.area_opts = NULL;
	cuda_host_to_device_mapper_args.dma = &aml_dma_cuda_host_to_device;
	cuda_host_to_device_mapper_args.dma_op = aml_dma_cuda_copy_1D;
	cuda_host_to_device_mapper_args.dma_op_arg = NULL;

	cuda_device_to_host_mapper_args.area = &aml_area_cuda;
	cuda_device_to_host_mapper_args.area_opts = NULL;
	cuda_device_to_host_mapper_args.dma = &aml_dma_cuda_device_to_host;
	cuda_device_to_host_mapper_args.dma_op = aml_dma_cuda_copy_1D;
	cuda_device_to_host_mapper_args.dma_op_arg = NULL;

	c = init_struct();
	__c = init_struct();

	// _c is the test value. We make sure it is different from
	// c, the original structure. Then we copy back c from
	// the cuda deepcopy __c into _c. Then we check c == __c.
	assert(aml_mapper_mmap(&struct_C_mapper,
	                       &cuda_host_to_device_mapper_args, c, &_c,
	                       1) == AML_SUCCESS);
	c->b[0].a->val = 4565467567;
	assert(!eq_struct(c, __c));

	// Deepcopy cuda copy back on host.
	if (aml_mapper_copy_back(&struct_C_mapper,
	                         &cuda_device_to_host_mapper_args, _c, c,
	                         1) != AML_SUCCESS)
		return 1;

	// Check for equality.
	assert(eq_struct(c, __c));

	// Cleanup
	aml_mapper_munmap(&struct_C_mapper, &cuda_device_to_host_mapper_args,
	                  _c);
	free(c);
	free(__c);
	aml_finalize();
	return 0;
}
