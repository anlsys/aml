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

#include "aml/area/cuda.h"
#include "aml/dma/cuda.h"

int aml_memcpy_cuda(struct aml_layout *dst,
                    const struct aml_layout *src,
                    void *arg)
{
	struct aml_dma_cuda_op_arg *op_arg = (struct aml_dma_cuda_op_arg *)arg;
	size_t size = (size_t)op_arg->op_arg;

	if (op_arg->data->kind == cudaMemcpyDeviceToDevice)
		return -AML_ENOTSUP;
	else {
		if (cudaMemcpyAsync(dst, src, size, op_arg->data->kind,
		                    op_arg->data->stream) != cudaSuccess)
			return -AML_FAILURE;
	}
	return AML_SUCCESS;
}

int main(int argc, char **argv)
{
	assert(aml_init(&argc, &argv) == AML_SUCCESS);
	if (!aml_support_backends(AML_BACKEND_CUDA))
		return 77;

	test_dma_memcpy(&aml_area_cuda, NULL, &aml_dma_cuda, aml_memcpy_cuda);

	test_dma_sync(&aml_area_cuda, NULL, &aml_dma_cuda, aml_memcpy_cuda);

	aml_finalize();
}
