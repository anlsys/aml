/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include "aml.h"

#include "aml/area/opencl.h"

static int aml_area_opencl_create_common(struct aml_area **area,
                                         cl_context context)
{
	struct aml_area *a;
	struct aml_area_opencl_data *data;

	a = (struct aml_area *)AML_INNER_MALLOC(struct aml_area,
	                                        struct aml_area_opencl_data);
	if (a == NULL)
		return -AML_ENOMEM;

	data = (struct aml_area_opencl_data *)AML_INNER_MALLOC_GET_FIELD(
	        a, 2, struct aml_area, struct aml_area_opencl_data);
	a->data = (struct aml_area_data *)data;
	data->context = context;

	*area = a;
	return AML_SUCCESS;
}

void aml_area_opencl_destroy(struct aml_area **area)
{
	if (area != NULL && *area != NULL) {
		free(*area);
		*area = NULL;
	}
}

//-----------------------------------------------------------------------------
// Buffer area
//-----------------------------------------------------------------------------

int aml_area_opencl_create(struct aml_area **area,
                           cl_context context,
                           const cl_mem_flags flags)
{
	struct aml_area *a;
	int err;
	struct aml_area_opencl_data *data;

	err = aml_area_opencl_create_common(&a, context);
	if (err != AML_SUCCESS)
		return err;
	a->ops = &aml_area_opencl_ops;
	data = (struct aml_area_opencl_data *)a->data;
	data->flags.buffer_flags = flags;
	*area = a;
	return AML_SUCCESS;
}

void *aml_area_opencl_mmap(const struct aml_area_data *area_data,
                           size_t size,
                           struct aml_area_mmap_options *options)
{
	cl_int cl_err;
	struct aml_area_opencl_data *d =
	        (struct aml_area_opencl_data *)area_data;
	cl_mem out;

	out = clCreateBuffer(d->context, d->flags.buffer_flags, size,
	                     (void *)options, &cl_err);

	switch (cl_err) {
	case CL_SUCCESS:
		break;
	case CL_INVALID_CONTEXT:
		assert(0);
	case CL_INVALID_VALUE:
	case CL_INVALID_HOST_PTR:
	case CL_INVALID_BUFFER_SIZE:
		aml_errno = -AML_EINVAL;
		return NULL;
	case CL_OUT_OF_RESOURCES:
	case CL_OUT_OF_HOST_MEMORY:
		aml_errno = -AML_ENOMEM;
		return NULL;
	case CL_MEM_OBJECT_ALLOCATION_FAILURE:
	default:
		aml_errno = -AML_FAILURE;
		return NULL;
	}

	// Risky. There is no guarantee cl_mem is smaller than a pointer.
	// It could be a large struct.
	return (void *)out;
}

int aml_area_opencl_munmap(const struct aml_area_data *area_data,
                           void *ptr,
                           const size_t size)
{
	(void)area_data;
	(void)size;
	cl_int cl_err = clReleaseMemObject((cl_mem)ptr);

	if (cl_err == CL_INVALID_MEM_OBJECT)
		return -AML_EINVAL;
	if (cl_err == CL_OUT_OF_RESOURCES || cl_err == CL_OUT_OF_HOST_MEMORY)
		return -AML_ENOMEM;
	return AML_SUCCESS;
}

struct aml_area_ops aml_area_opencl_ops = {
        .mmap = aml_area_opencl_mmap,
        .munmap = aml_area_opencl_munmap,
        .fprintf = NULL,
};

//-----------------------------------------------------------------------------
// SVM area
//-----------------------------------------------------------------------------

int aml_area_opencl_svm_create(struct aml_area **area,
                               cl_context context,
                               const cl_svm_mem_flags flags,
                               cl_uint alignement)
{
	struct aml_area *a;
	int err;
	struct aml_area_opencl_data *data;

	err = aml_area_opencl_create_common(&a, context);
	if (err != AML_SUCCESS)
		return err;
	a->ops = &aml_area_opencl_svm_ops;
	data = (struct aml_area_opencl_data *)a->data;
	data->flags.svm_flags.flags = flags;
	data->flags.svm_flags.alignement = alignement;

	*area = a;
	return AML_SUCCESS;
}

void *aml_area_opencl_svm_mmap(const struct aml_area_data *area_data,
                               size_t size,
                               struct aml_area_mmap_options *options)
{
	(void)options;
	struct aml_area_opencl_data *data =
	        (struct aml_area_opencl_data *)area_data;
	return clSVMAlloc(data->context, data->flags.svm_flags.flags, size,
	                  data->flags.svm_flags.alignement);
}

int aml_area_opencl_svm_munmap(const struct aml_area_data *area_data,
                               void *ptr,
                               const size_t size)
{
	(void)size;
	struct aml_area_opencl_data *data =
	        (struct aml_area_opencl_data *)area_data;
	clSVMFree(data->context, ptr);
	return AML_SUCCESS;
}

struct aml_area_ops aml_area_opencl_svm_ops = {
        .mmap = aml_area_opencl_svm_mmap,
        .munmap = aml_area_opencl_svm_munmap,
        .fprintf = NULL,
};
