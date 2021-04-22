/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *******************************************************************************/

#include "config.h"

#include <stdlib.h>
#include <string.h>

#include "aml.h"

#include "aml/area/opencl.h"

const size_t size = 1 << 20;

void test_device_mmap(cl_context context, cl_command_queue queue)
{
	void *host_data;
	void *host_copy;
	void *device_data;
	struct aml_area *area;

	// Create area on device that will copy on map.
	assert(!aml_area_opencl_create(
	        &area, context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR));

	// Make host and device buffers and initialize one host buffer.
	host_data = malloc(size);
	assert(host_data != NULL);
	memset(host_data, 0, size);
	host_copy = malloc(size);
	assert(host_copy != NULL);
	memset(host_copy, 1, size);
	device_data = aml_area_mmap(area, size, host_data);
	assert(device_data);

	// Initialized host buffer is supposed to be copied into device buffer.
	// Copy it back into the check buffer.
	assert(clEnqueueReadBuffer(queue, device_data, CL_TRUE, 0, size,
	                           host_copy, 0, NULL, NULL) == CL_SUCCESS);
	assert(!memcmp(host_data, host_copy, size));

	// Cleanup
	assert(aml_area_munmap(area, device_data, size) == AML_SUCCESS);
	free(host_data);
	free(host_copy);
	aml_area_opencl_destroy(&area);
}

void test_host_mmap(cl_context context)
{
	void *host_data;
	void *host_copy;
	struct aml_area *area;

	// Create area on device that will copy on map.
	assert(!aml_area_opencl_create(&area, context,
	                               CL_MEM_READ_WRITE |
	                                       CL_MEM_ALLOC_HOST_PTR |
	                                       CL_MEM_COPY_HOST_PTR));

	host_data = malloc(size);
	assert(host_data != NULL);
	memset(host_data, 1, size);
	host_copy = aml_area_mmap(area, size, host_data);
	assert(host_copy);
	assert(!memcmp(host_data, host_copy, size));

	assert(aml_area_munmap(area, host_copy, size) == AML_SUCCESS);
	free(host_data);
	aml_area_opencl_destroy(&area);
}

void test_svm_mmap(cl_context context, cl_command_queue queue)
{
	void *ptr = NULL;
	void *dst = NULL;
	struct aml_area *area;

	// Set test buffer
	dst = malloc(size);
	assert(dst != NULL);
	memset(dst, 1, size);

	// Create svm area.
	assert(aml_area_opencl_svm_create(&area, context, CL_MEM_READ_WRITE,
	                                  0) == AML_SUCCESS);
	ptr = aml_area_mmap(area, size, NULL);
	assert(ptr);

	// Set SVM pointer to different value.
	assert(clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_WRITE, ptr, size, 0, NULL,
	                       NULL) == CL_SUCCESS);
	memset(ptr, 0, size);
	assert(clEnqueueSVMUnmap(queue, ptr, 0, NULL, NULL) == CL_SUCCESS);

	// Copy SVM region into dst.
	assert(clEnqueueSVMMemcpy(queue, CL_TRUE, dst, ptr, size, 0, NULL,
	                          NULL) == CL_SUCCESS);

	// Check if the result SVM region and dst match
	assert(!memcmp(dst, ptr, size));

	assert(aml_area_munmap(area, ptr, size) == AML_SUCCESS);
	aml_area_opencl_destroy(&area);
}

//-----------------------------------------------------------------------------

cl_int cl_setup(cl_platform_id platform,
                const cl_device_type device_type,
                cl_context *context,
                cl_command_queue *q)
{
	cl_int err;
	cl_device_id device;

	err = clGetDeviceIDs(platform, device_type, 1, &device, NULL);
	if (err != AML_SUCCESS)
		return err;

	cl_context_properties properties[3] = {CL_CONTEXT_PLATFORM,
	                                       (cl_context_properties)platform,
	                                       (cl_context_properties)0};

	if (context != NULL) {
		*context = clCreateContext(properties, 1, &device, NULL, NULL,
		                           &err);
		if (err != CL_SUCCESS)
			return err;
	}

	if (q != NULL) {
		*q = clCreateCommandQueueWithProperties(*context, device, NULL,
		                                        &err);
		if (err != CL_SUCCESS)
			return err;
	}

	return CL_SUCCESS;
}

void cl_cleanup(cl_context *context, cl_command_queue *q)
{

	if (context != NULL)
		assert(clReleaseContext(*context) == CL_SUCCESS);

	if (q != NULL)
		assert(clReleaseCommandQueue(*q) == CL_SUCCESS);
}

//-----------------------------------------------------------------------------

int main(void)
{
	cl_context context;
	cl_command_queue queue;
	cl_uint num_platforms = 16;
	cl_platform_id platforms[num_platforms];

	if (!aml_support_backends(AML_BACKEND_OPENCL))
		return 77;

	assert(clGetPlatformIDs(num_platforms, platforms, &num_platforms) ==
	       CL_SUCCESS);

	for (cl_uint i = 0; i < num_platforms; i++) {
		// CPU tests
		if (cl_setup(platforms[i], CL_DEVICE_TYPE_CPU, &context,
		             &queue) == CL_SUCCESS) {
			test_device_mmap(context, queue);
			test_host_mmap(context);
			test_svm_mmap(context, queue);
			cl_cleanup(&context, &queue);
		}

		// ACCELERATOR tests
		if (cl_setup(platforms[i], CL_DEVICE_TYPE_ACCELERATOR, &context,
		             &queue) == CL_SUCCESS) {
			test_device_mmap(context, queue);
			test_svm_mmap(context, queue);
			cl_cleanup(&context, &queue);
		}

		// GPU tests
		if (cl_setup(platforms[i], CL_DEVICE_TYPE_GPU, &context,
		             &queue) == CL_SUCCESS) {
			test_device_mmap(context, queue);
			test_svm_mmap(context, queue);
			cl_cleanup(&context, &queue);
		}
	}
	return 0;
}
