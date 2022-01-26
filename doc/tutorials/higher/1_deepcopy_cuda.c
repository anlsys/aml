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
	struct C *c;
	aml_deepcopy_data device_c, host_c;

	assert(aml_init(&argc, &argv) == AML_SUCCESS);
	if (!aml_support_backends(AML_BACKEND_CUDA))
		return 77;

	c = init_struct();

	// Deepcopy to device.
	assert(aml_mapper_deepcopy(&device_c, (void *)c, &struct_C_mapper,
	                           &aml_area_cuda, NULL, NULL, &aml_dma_cuda,
	                           NULL,
	                           aml_dma_cuda_memcpy_op) == AML_SUCCESS);

	// Deepcopy from device to host
	assert(aml_mapper_deepcopy(&host_c, aml_deepcopy_ptr(device_c),
	                           &struct_C_mapper, &aml_area_linux, NULL,
	                           &aml_dma_cuda, &aml_dma_cuda,
	                           aml_dma_cuda_memcpy_op,
	                           aml_dma_cuda_memcpy_op) == AML_SUCCESS);

	// Check for equality.
	assert(eq_struct(c, (struct C *)aml_deepcopy_ptr(host_c)));

	// Cleanup
	aml_mapper_deepfree(device_c);
	aml_mapper_deepfree(host_c);
	free(c);
	aml_finalize();
	return 0;
}
