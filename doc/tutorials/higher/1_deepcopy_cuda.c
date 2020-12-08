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

int main(int argc, char **argv)
{
	size_t size;
	struct C *c, *_c, *__c;

	assert(aml_init(&argc, &argv) == AML_SUCCESS);
	if (!aml_support_backends(AML_BACKEND_CUDA))
		return 77;

	c = init_struct();

	// _c is the test value. We make sure it is different from
	// c, the original structure. Then we copy back c from
	// the cuda deepcopy __c into _c. Then we check c == __c.
	_c = aml_mapper_mmap(&struct_C_mapper, c, &aml_area_linux, NULL,
	                     aml_dma_linux_sequential, NULL, NULL, &size);
	_c->b[0].a->val = 4565467567;
	assert(!eq_struct(c, _c));

	// Mapping on cuda area.
	__c = aml_mapper_mmap(&struct_C_mapper, c, &aml_area_cuda, NULL,
	                      &aml_dma_cuda_host_to_device,
	                      aml_dma_cuda_copy_1D, NULL, &size);

	// Deepcopy cuda copy back on host.
	if (aml_mapper_copy_back(&struct_C_mapper, __c, _c,
	                         &aml_dma_cuda_device_to_host,
	                         aml_dma_cuda_copy_1D, NULL) != AML_SUCCESS)
		return 1;

	// Check for equality.
	assert(eq_struct(c, _c));

	// Cleanup
	aml_area_munmap(&aml_area_cuda, __c, size);
	aml_area_munmap(&aml_area_linux, _c, size);
	free(c);
	aml_finalize();
	return 0;
}
