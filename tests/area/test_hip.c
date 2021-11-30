/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include "config.h"

#include <hip/hip_runtime_api.h>
#include <stdlib.h>
#include <string.h>

#include "aml.h"

#include "aml/area/hip.h"

const size_t sizes[4] = {1, 32, 4096, 1 << 20};

void test_device_mmap(const int device)
{
	void *host_data;
	void *host_copy;
	void *device_data;
	struct aml_area *area;
	size_t ns = sizeof(sizes) / sizeof(*sizes);
	int err;
	size_t size;

	assert(!aml_area_hip_create(&area, device, AML_AREA_HIP_FLAG_DEFAULT));
	host_data = malloc(sizes[ns - 1]);
	assert(host_data != NULL);
	host_copy = malloc(sizes[ns - 1]);
	assert(host_copy != NULL);

	for (size_t i = 0; i < ns; i++) {
		size = sizes[i];
		memset(host_data, 0, size);
		memset(host_copy, 1, size);

		device_data = aml_area_mmap(area, size, NULL);
		assert(device_data);
		err = hipMemcpy(device_data, host_data, size,
		                hipMemcpyHostToDevice);
		assert(err == hipSuccess);
		err = hipMemcpy(host_copy, device_data, size,
		                hipMemcpyDeviceToHost);
		assert(err == hipSuccess);
		assert(!memcmp(host_data, host_copy, size));
		assert(aml_area_munmap(area, device_data, size) == AML_SUCCESS);
	}

	free(host_data);
	free(host_copy);
	aml_area_hip_destroy(&area);
}

void test_host_mmap(const int device)
{
	void *host_data;
	void *host_copy;
	struct aml_area *area;
	size_t ns = sizeof(sizes) / sizeof(*sizes);
	size_t size;

	assert(!aml_area_hip_create(&area, device,
	                            AML_AREA_HIP_FLAG_ALLOC_HOST));
	host_copy = malloc(sizes[ns - 1]);
	assert(host_copy != NULL);

	for (size_t i = 0; i < ns; i++) {
		size = sizes[i];
		memset(host_copy, 1, size);
		host_data = aml_area_mmap(area, size, NULL);
		assert(host_data);
		memcpy(host_data, host_copy, size);
		assert(!memcmp(host_data, host_copy, size));
		assert(aml_area_munmap(area, host_data, size) == AML_SUCCESS);
	}

	free(host_copy);
	aml_area_hip_destroy(&area);
}

void test_mapped_mmap(const int device)
{
	void *host_data;
	void *host_copy;
	void *device_data;
	struct aml_area *area;
	size_t ns = sizeof(sizes) / sizeof(*sizes);
	size_t size;
	struct aml_area_hip_mmap_options options = {
	        .device = device,
	        .ptr = NULL,
	};

	// Data initialization
	host_copy = malloc(sizes[ns - 1]);
	assert(host_copy != NULL);

	// Map existing host data.
	host_data = malloc(sizes[ns - 1]);
	assert(host_data != NULL);
	assert(!aml_area_hip_create(&area, device,
	                            AML_AREA_HIP_FLAG_ALLOC_MAPPED));
	options.ptr = host_data;
	for (size_t i = 0; i < ns; i++) {
		size = sizes[i];
		assert(!aml_area_mmap(
		        area, size, (struct aml_area_mmap_options *)&options));
		assert(!hipHostGetDevicePointer(&device_data, host_data, 0));
		assert(device_data);

		memset(host_data, 0, size);
		memset(host_copy, 1, size);
		assert(!hipMemcpy(host_copy, device_data, size,
		                  hipMemcpyDeviceToHost));
		assert(!memcmp(host_data, host_copy, size));

		memset(host_data, 0, size);
		memset(host_copy, 1, size);
		assert(!hipMemcpy(host_copy, device_data, size,
		                  hipMemcpyHostToDevice));
		assert(!memcmp(host_data, host_copy, size));

		assert(!aml_area_munmap(area, host_data, size));
	}
	free(host_data);
	aml_area_hip_destroy(&area);

	// Map new host data.
	assert(!aml_area_hip_create(&area, device,
	                            AML_AREA_HIP_FLAG_ALLOC_MAPPED |
	                                    AML_AREA_HIP_FLAG_ALLOC_HOST));
	options.ptr = NULL;
	for (size_t i = 0; i < ns; i++) {
		size = sizes[i];
		host_data = aml_area_mmap(
		        area, size, (struct aml_area_mmap_options *)&options);
		assert(host_data != NULL);
		assert(!hipHostGetDevicePointer(&device_data, host_data, 0));
		assert(device_data);

		memset(host_data, 0, size);
		memset(host_copy, 1, size);
		assert(!hipMemcpy(host_copy, device_data, size,
		                  hipMemcpyDeviceToHost));
		assert(!memcmp(host_data, host_copy, size));

		memset(host_data, 0, size);
		memset(host_copy, 1, size);
		assert(!hipMemcpy(host_copy, device_data, size,
		                  hipMemcpyHostToDevice));
		assert(!memcmp(host_data, host_copy, size));

		assert(!aml_area_munmap(area, host_data, size));
	}
	free(host_copy);
	aml_area_hip_destroy(&area);
}

void test_unified_mmap(const int device)
{
	void *unified_data;
	void *host_copy;
	struct aml_area *area;
	size_t ns = sizeof(sizes) / sizeof(*sizes);
	size_t size;

	// Data initialization
	host_copy = malloc(sizes[ns - 1]);
	assert(host_copy != NULL);

	// Map existing host data.
	assert(!aml_area_hip_create(&area, device,
	                            AML_AREA_HIP_FLAG_ALLOC_UNIFIED));
	for (size_t i = 0; i < ns; i++) {
		size = sizes[i];
		unified_data = aml_area_mmap(area, size, NULL);
		assert(unified_data != NULL);

		memset(unified_data, 0, size);
		memset(host_copy, 1, size);
		hipDeviceSynchronize();
		assert(!hipMemcpy(host_copy, unified_data, size,
		                  hipMemcpyDeviceToHost));
		assert(!memcmp(unified_data, host_copy, size));

		memset(unified_data, 0, size);
		memset(host_copy, 1, size);
		hipDeviceSynchronize();
		assert(!hipMemcpy(unified_data, host_copy, size,
		                  hipMemcpyHostToDevice));
		assert(!memcmp(unified_data, host_copy, size));

		assert(!aml_area_munmap(area, unified_data, size));
	}

	aml_area_hip_destroy(&area);
	free(host_copy);
}

int main(void)
{
	int num_devices;
	unsigned int flags;
	int has_device_map;
	int has_unified_mem;
	int has_register_ptr;
	int current_device;

	if (!aml_support_backends(AML_BACKEND_HIP))
		return 77;

	assert(hipGetDeviceCount(&num_devices) == hipSuccess);
	assert(hipGetDevice(&current_device) == hipSuccess);

	for (int i = 0; i < num_devices; i++) {
		// check device features
		assert(!hipSetDevice(i));
		assert(!hipGetDeviceFlags(&flags));
		has_device_map = flags & hipDeviceMapHost;
		assert(!hipSetDevice(current_device));
		assert(!hipDeviceGetAttribute(
		        &has_unified_mem, hipDeviceAttributeManagedMemory, i));
		// Hip does not declare any equivalent to
		// "cudaDevAttrCanUseHostPointerForRegisteredMem"
		// assert(!hipDeviceGetAttribute(
		//         &has_register_ptr,
		//         hipDevAttrCanUseHostPointerForRegisteredMem, i));

		test_device_mmap(i);
		test_host_mmap(i);
		if (has_device_map && has_register_ptr)
			test_mapped_mmap(i);
		if (has_unified_mem)
			test_unified_mmap(i);
	}

	return 0;
}
