/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <aml.h>
#include <aml/area/cuda.h>

void test_default_area(const size_t size, void *host_buf)
{
	void *device_buf;

	// Map data on current device.
	device_buf = aml_area_mmap(&aml_area_cuda, size, NULL);
	if (device_buf == NULL) {
		aml_perror("aml_area_cuda");
		exit(1);
	}
	// Check we can perform a data transfer to mapped device memory.
	assert(cudaMemcpy(device_buf,
			  host_buf,
			  size, cudaMemcpyHostToDevice) == cudaSuccess);

	printf("Default cuda area worked!\n");

	// Cleanup
	aml_area_munmap(&aml_area_cuda, device_buf, size);
}

void test_custom_area(const size_t size, void *host_buf)
{
	int err;
	void *device_buf;
	struct aml_area *area;

	// When calling mmap, we are going to specify a device
	// and host side memory to map to.
	// We want to map host_buf to device_buf.
	struct aml_area_cuda_mmap_options opts = {
		.device = 0,
		.ptr = host_buf
	};

	// Create an area that will map data on device 0, with host memory.
	err = aml_area_cuda_create(&area, 0, AML_AREA_CUDA_FLAG_ALLOC_MAPPED);
	if (err != AML_SUCCESS) {
		fprintf(stderr, "aml_area_cuda_create: %s\n",
			aml_strerror(err));
		exit(1);
	}
	// Map host memory with device memory.
	if (aml_area_mmap(area, size,
			  (struct aml_area_mmap_options *)&opts) == NULL) {
		aml_perror("aml_area_cuda_create");
		exit(1);
	}
	// Get device memory that is mapped with host memory.
	assert(cudaHostGetDevicePointer(&device_buf, host_buf, 0) ==
	       cudaSuccess);

	// Set data from host.
	memset(host_buf, '#', size);

	// Check that data on device has been set to the same value,
	// i.e., mapping works.
	assert(cudaMemcpy(host_buf,
			  device_buf,
			  size, cudaMemcpyDeviceToHost) == cudaSuccess);
	for (size_t i = 0; i < size; i++)
		assert(((char *)host_buf)[i] == '#');

	printf("Custom cuda area worked!\n");

	// Cleanup
	aml_area_munmap(area, device_buf, size);
	aml_area_cuda_destroy(&area);
}

int main(void)
{
	// Skip tutorial if this is not supported.
	if (!aml_support_backends(AML_BACKEND_CUDA))
		return 77;

	const size_t size = (2 << 16);	// 16 pages
	void *host_buf = malloc(size);
	if (host_buf == NULL)
		return 1;

	test_default_area(size, host_buf);
	test_custom_area(size, host_buf);

	free(host_buf);
	return 0;
}
