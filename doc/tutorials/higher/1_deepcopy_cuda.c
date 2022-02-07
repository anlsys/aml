/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

//- Keep previous tutorial ----------------------------------------------------

#define DEEP_COPY_CUDA
#include "0_deepcopy_linux.c"

//- Cuda specific allocation / data movement ----------------------------------

#include "aml/area/cuda.h"
#include "aml/dma/cuda.h"

int main(int argc, char **argv)
{
	struct C *c, *host_c;
	void *device_c;
	aml_mapped_ptrs d0, d1;

	assert(aml_init(&argc, &argv) == AML_SUCCESS);
	if (!aml_support_backends(AML_BACKEND_CUDA))
		return 77;

	c = init_struct();

	// Deepcopy to device.
	device_c = aml_mapper_deepcopy(&d0, (void *)c, &struct_C_mapper,
	                               &aml_area_cuda, NULL, NULL,
	                               &aml_dma_cuda, NULL,
	                               aml_dma_cuda_memcpy_op);

	// Deepcopy from device to host
	host_c = aml_mapper_deepcopy(&d1, device_c, &struct_C_mapper,
	                             &aml_area_linux, NULL, &aml_dma_cuda,
	                             &aml_dma_cuda, aml_dma_cuda_memcpy_op,
	                             aml_dma_cuda_memcpy_op);

	// Check for equality.
	assert(eq_struct(c, host_c));

	// Cleanup
	aml_mapper_deepfree(d0);
	aml_mapper_deepfree(d1);
	free(c);
	aml_finalize();
	return 0;
}
