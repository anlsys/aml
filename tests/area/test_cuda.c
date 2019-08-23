/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *******************************************************************************/
#include "aml.h"
#include "config.h"
#include "aml/area/cuda.h"
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>

const size_t sizes[4] = {1, 32, 4096, 1<<20};

void test_mmap(const struct aml_area *area, const size_t size)
{
	int err;
	void *host_data;
	void *host_copy;
	void *device_data;
        int flags;

	flags = ((struct aml_area_cuda_data *)area->data)->flags;
	
	host_data = malloc(size);
	assert(host_data);
	memset(host_data, 1, size);

	host_copy = malloc(size);
	assert(host_copy);
	memset(host_copy, 0, size);

	// Test standalone GPU side allocation.
	device_data = aml_area_mmap(area, NULL, size);
	assert(device_data);
	err = cudaMemcpy(device_data, host_data, size, cudaMemcpyHostToDevice);
	assert(err == cudaSuccess);
	err = cudaMemcpy(host_copy, device_data, size, cudaMemcpyDeviceToHost);
	assert(err == cudaSuccess);
	assert(!memcmp(host_data, host_copy, size));	
	assert(aml_area_munmap(area, device_data, size) == AML_SUCCESS);

	// Test memory mapped allocation
	memset(host_copy, 0, size);
	device_data = aml_area_mmap(area, host_data, size);
	assert(device_data);
	err = cudaMemcpy(host_copy, device_data, size, cudaMemcpyDeviceToHost);
	assert(err == cudaSuccess);
	assert(!memcmp(host_data, host_copy, size));
	assert(aml_area_munmap(area, device_data, size) == AML_SUCCESS);
	free(host_data);
	free(host_copy);
}

void test_area(const struct aml_area *area)
{
	for(size_t i = 0; i < sizeof(sizes)/sizeof(*sizes); i++){
		test_mmap(area, sizes[i]);
	}
}

int main(){
	int num_devices;
	int aml_error;
	struct aml_area *area_cuda;

	assert(cudaGetDeviceCount(&num_devices) == cudaSuccess);

	for(int i = -1; i<num_devices; i++){
		aml_error = aml_area_cuda_create(&area_cuda, i, AML_AREA_CUDA_ATTACH_GLOBAL);
		assert(aml_error == AML_SUCCESS);
		test_area(area_cuda);		
	        aml_area_cuda_destroy(&area_cuda);
		
		aml_error = aml_area_cuda_create(&area_cuda, i, AML_AREA_CUDA_ATTACH_HOST);
		assert(aml_error == AML_SUCCESS);
		test_area(area_cuda);		
	        aml_area_cuda_destroy(&area_cuda);
	}

	return 0;
}
