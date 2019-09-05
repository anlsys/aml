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

void test_device_mmap(const int device)
{
	void *host_data;
	void *host_copy;
	void *device_data;
	struct aml_area *area;
	int ns = sizeof(sizes) / sizeof(*sizes);
	int err;
	size_t size;

        assert( aml_area_cuda_create(&area, device, AML_AREA_CUDA_FLAG_DEFAULT) == AML_SUCCESS );
	host_data = malloc(sizes[ns-1]);
	assert(host_data != NULL);
        host_copy = malloc(sizes[ns-1]);
	assert(host_copy != NULL);
	
	
	for(size_t i=0; i<ns; i++){
		size = sizes[i];
		memset(host_data, 0, size);
		memset(host_copy, 1, size);
		
		device_data = aml_area_mmap(area, size, NULL);
		assert(device_data);
		err = cudaMemcpy(device_data, host_data, size, cudaMemcpyHostToDevice);
		assert(err == cudaSuccess);
		err = cudaMemcpy(host_copy, device_data, size, cudaMemcpyDeviceToHost);
		assert(err == cudaSuccess);
		assert(!memcmp(host_data, host_copy, size));	
		assert(aml_area_munmap(area, device_data, size) == AML_SUCCESS);		
	}
	
	free(host_data);
	free(host_copy);
	aml_area_cuda_destroy(&area);
}

void test_host_mmap(const int device)
{
	void *host_data;
	void *host_copy;
	struct aml_area *area;
	int ns = sizeof(sizes) / sizeof(*sizes);
	size_t size;

        assert( aml_area_cuda_create(&area, device, AML_AREA_CUDA_FLAG_ALLOC_HOST) == AML_SUCCESS );
        host_copy = malloc(sizes[ns-1]);
	assert(host_copy != NULL);
	
	for(size_t i=0; i<ns; i++){
		size = sizes[i];
		memset(host_copy, 1, size);		
		host_data = aml_area_mmap(area, size, NULL);
		assert(host_data);
		memcpy(host_data, host_copy, size);
		assert(!memcmp(host_data, host_copy, size));
		assert(aml_area_munmap(area, host_data, size) == AML_SUCCESS);		
	}
	
	free(host_copy);
	aml_area_cuda_destroy(&area);
}

void test_mapped_mmap(const int device)
{
	void *host_data;
	void *host_copy;
	void *device_data;
	struct aml_area *area;
	int ns = sizeof(sizes) / sizeof(*sizes);
	size_t size;
	struct aml_area_cuda_mmap_options options = { .device = device, .ptr = NULL, };

	// Data initialization
        host_copy = malloc(sizes[ns-1]);
	assert(host_copy != NULL);
	
	// Map existing host data.
	host_data = malloc(sizes[ns-1]);
	assert(host_data != NULL);
        assert( aml_area_cuda_create(&area, device, AML_AREA_CUDA_FLAG_ALLOC_MAPPED) == AML_SUCCESS );
	options.ptr = host_data;	
	for(size_t i=0; i<ns; i++){
		size = sizes[i];
		assert(aml_area_mmap(area, size, (struct aml_area_mmap_options *)&options) != NULL);
		assert(cudaHostGetDevicePointer(&device_data, host_data, 0) == cudaSuccess);
		assert(device_data);

		memset(host_data, 0, size);
		memset(host_copy, 1, size);				
	        assert(cudaMemcpy(host_copy, device_data, size, cudaMemcpyDeviceToHost) == cudaSuccess);
		assert(!memcmp(host_data, host_copy, size));

		memset(host_data, 0, size);
		memset(host_copy, 1, size);				
		assert(cudaMemcpy(host_copy, device_data, size, cudaMemcpyHostToDevice) == cudaSuccess);
		assert(!memcmp(host_data, host_copy, size));
		
		assert(aml_area_munmap(area, host_data, size) == AML_SUCCESS);		
	}
	free(host_data);
	aml_area_cuda_destroy(&area);

	// Map new host data.
	assert( aml_area_cuda_create(&area,
				     device,
				     AML_AREA_CUDA_FLAG_ALLOC_MAPPED |
				     AML_AREA_CUDA_FLAG_ALLOC_HOST) == AML_SUCCESS );
	options.ptr = NULL;
	for(size_t i=0; i<ns; i++){
		size = sizes[i];
		host_data = aml_area_mmap(area, size, (struct aml_area_mmap_options *)&options);
		assert(host_data != NULL);
		assert(cudaHostGetDevicePointer(&device_data, host_data, 0) == cudaSuccess);
		assert(device_data);

		memset(host_data, 0, size);
		memset(host_copy, 1, size);				
		assert(cudaMemcpy(host_copy, device_data, size, cudaMemcpyDeviceToHost) == cudaSuccess);
		assert(!memcmp(host_data, host_copy, size));

		memset(host_data, 0, size);
		memset(host_copy, 1, size);				
		assert(cudaMemcpy(host_copy, device_data, size, cudaMemcpyHostToDevice) == cudaSuccess);
		assert(!memcmp(host_data, host_copy, size));

		assert(aml_area_munmap(area, host_data, size) == AML_SUCCESS);		
	}
	free(host_copy);
	aml_area_cuda_destroy(&area);
}

void test_unified_mmap(const int device)
{
	void *unified_data;
	void *host_copy;
	struct aml_area *area;
	int ns = sizeof(sizes) / sizeof(*sizes);
	size_t size;

	// Data initialization
        host_copy = malloc(sizes[ns-1]);
	assert(host_copy != NULL);
	
	// Map existing host data.
        assert( aml_area_cuda_create(&area, device, AML_AREA_CUDA_FLAG_ALLOC_UNIFIED) == AML_SUCCESS );
	for(size_t i=0; i<ns; i++){
		size = sizes[i];
	        unified_data = aml_area_mmap(area, size, NULL);
		assert(unified_data != NULL);
		
		memset(unified_data, 0, size);
		memset(host_copy, 1, size);
		cudaDeviceSynchronize();
	        assert(cudaMemcpy(host_copy, unified_data, size, cudaMemcpyDeviceToHost) == cudaSuccess);
		assert(!memcmp(unified_data, host_copy, size));

		memset(unified_data, 0, size);
		memset(host_copy, 1, size);
		cudaDeviceSynchronize();
		assert(cudaMemcpy(unified_data, host_copy, size, cudaMemcpyHostToDevice) == cudaSuccess);
		assert(!memcmp(unified_data, host_copy, size));
		
		assert(aml_area_munmap(area, unified_data, size) == AML_SUCCESS);
	}
	free(host_copy);
}

int main(){
	int num_devices;
	int flags;
	int has_device_map;
	int has_unified_mem;
	int has_register_ptr;
	int current_device;

	assert(cudaGetDeviceCount(&num_devices) == cudaSuccess);
	assert(cudaGetDevice(&current_device) == cudaSuccess);
	
	for(int i = 0; i<num_devices; i++){
		// check device features
		assert(cudaSetDevice(i) == cudaSuccess);
		assert(cudaGetDeviceFlags(&flags) == cudaSuccess);
		has_device_map = flags & cudaDeviceMapHost;
		assert(cudaSetDevice(current_device) == cudaSuccess);
		assert(cudaDeviceGetAttribute(&has_unified_mem,
					      cudaDevAttrManagedMemory,
					      i) == cudaSuccess);		
		assert(cudaDeviceGetAttribute(&has_register_ptr,
					      cudaDevAttrCanUseHostPointerForRegisteredMem,
					      i) == cudaSuccess);
		
		test_device_mmap(i);
		test_host_mmap(i);
		if ( has_device_map && has_register_ptr )
			test_mapped_mmap(i);
		if ( has_unified_mem )
			test_unified_mmap(i);
	}

	return 0;
}