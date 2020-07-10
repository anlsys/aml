/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include <cuda.h>
#include <cuda_runtime.h>

#include "aml.h"

#include "aml/dma/cuda.h"
#include "aml/layout/cuda.h"
#include "aml/layout/dense.h"

// Data
#define size (1 << 24) // 16 MB
void *host_data;
void *device_data;

// Layout
const size_t ndims = 1;
const size_t dims = size;
const size_t element_size = 1;
const size_t stride = 0;
const size_t pitch = 1;
struct aml_layout *host_layout;
struct aml_layout *device_layout;

// Dma
struct aml_dma *host_device;
struct aml_dma *device_host;

void setup()
{
	// Data
	host_data = malloc(size);
	assert(host_data != NULL);
	assert(cudaMalloc(&device_data, size) == cudaSuccess);

	// Layout
	assert(aml_layout_dense_create(&host_layout, host_data,
	                               AML_LAYOUT_ORDER_COLUMN_MAJOR,
	                               element_size, ndims, &dims, &stride,
	                               &pitch) == AML_SUCCESS);
	assert(aml_layout_cuda_create(&device_layout, device_data, 0,
	                              element_size,
	                              AML_LAYOUT_ORDER_COLUMN_MAJOR, ndims,
	                              &dims, &stride, &pitch) == AML_SUCCESS);

	// Dma
	assert(aml_dma_cuda_create(&host_device, cudaMemcpyHostToDevice) ==
	       AML_SUCCESS);
	assert(aml_dma_cuda_create(&device_host, cudaMemcpyDeviceToHost) ==
	       AML_SUCCESS);
}

void teardown()
{
	// Data
	free(host_data);
	cudaFree(device_data);

	// Layout
	aml_layout_dense_destroy(&device_layout);
	aml_layout_cuda_destroy(&device_layout);

	// Dma
	aml_dma_cuda_destroy(&host_device);
	aml_dma_cuda_destroy(&device_host);
}

// Fill host_data and device_data
void initialize_buffers(int host_val, int device_val)
{
	memset(host_data, host_val, size);
	assert(cudaMemset(device_data, device_val, size) == cudaSuccess);
}

int main(int argc, char **argv)
{
	aml_init(&argc, &argv);

	if (!aml_support_backends(AML_BACKEND_CUDA))
		return 77;
	setup();

	void *check = malloc(size);

	// Dma from device to host.
	initialize_buffers(1, 0);
	assert(aml_dma_copy_custom(device_host, host_layout, device_layout,
	                           aml_dma_cuda_copy_1D, NULL) == AML_SUCCESS);
	memset(check, 0, size);
	assert(!memcmp(check, host_data, size));

	// Dma from host to device.
	initialize_buffers(1, 0);
	assert(aml_dma_copy_custom(host_device, device_layout, host_layout,
	                           aml_dma_cuda_copy_1D, NULL) == AML_SUCCESS);
	assert(cudaMemcpy(check, device_data, size, cudaMemcpyDeviceToHost) ==
	       cudaSuccess);
	assert(!memcmp(check, host_data, size));

	free(check);
	teardown();
	aml_finalize();
}
