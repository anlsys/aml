/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *******************************************************************************/

#ifndef AML_AREA_OPENCL_H
#define AML_AREA_OPENCL_H

#include <CL/opencl.h>

/**
 * @defgroup aml_area_opencl "AML OpenCL Areas"
 * @brief OpenCL Implementation of Areas.
 * @code
 * #include <aml/area/opencl.h>
 * @endcode
 *
 * OpenCL implementation of AML areas.
 * This building block relies on OpenCL implementation of
 * device memory allocation to provide mmap/munmap on device memory.
 * Additional documentation of OpenCL memory model can be found here:
 * @see
 *https://www.khronos.org/registry/OpenCL/specs/2.2/html/OpenCL_API.html#_memory_model
 *
 * @{
 **/

/** Flags and parameter passed to clSVMalloc. **/
struct aml_area_opencl_svm_flags {
	cl_svm_mem_flags flags;
	cl_uint alignement;
};

/** Implementation of aml_area_data for cuda areas. **/
struct aml_area_opencl_data {
	/** Embed platforms ID, devices ID, devices properties */
	cl_context context;
	/** flags used in aml_area_opencl_data to map data **/
	union aml_area_opencl_flags {
		cl_mem_flags buffer_flags;
		struct aml_area_opencl_svm_flags svm_flags;
	} flags;
};

/** aml OpenCL area hooks (no SVM). **/
extern struct aml_area_ops aml_area_opencl_ops;

/**
 * This will spawn areas mapping device side or host side only memory.
 * This area creator does not allow mapping of memory regions where OpenCL is
 * responsible for synchronization with other APIs.
 * @see
 * https://www.khronos.org/registry/OpenCL/specs/2.2/html/OpenCL_API.html#context-properties-table
 * @param[out] area: A pointer to the area to allocate.
 * @param[in] context: A valid OpenCL context.
 * @param[in] flags: Flags to pass to `clCreateBuffer()`.
 * @see
 * https://www.khronos.org/registry/OpenCL//sdk/2.1/docs/man/xhtml/clCreateBuffer.html
 * @return AML_SUCCESS on success.
 * @return -AML_ENOMEM if there were not enough memory available to allocate
 * area.
 */
int aml_area_opencl_create(struct aml_area **area,
                           cl_context context,
                           const cl_mem_flags flags);

/**
 * OpenCL implementation of mmap operation for aml area created with
 * `aml_area_opencl_create()`.
 * @param[in] area_data: Area data of type `struct aml_area_opencl_data`
 * where flag field contains `cl_mem_flags`.
 * @param[in] size: The size to allocate.
 * @param[in] options: NULL or host pointer if area flags contain
 * CL_MEM_USE_HOST_PTR or CL_MEM_ALLOC_HOST_PTR or CL_MEM_COPY_HOST_PTR.
 * @see aml_area
 * @return A valid `cl_mem` casted into `(void *)` on success.
 * @return NULL on error with aml_errno set to:
 * *  -AML_EINVAL if size is 0, area flags are not valid or options is NULL
 * while flags specify use of host pointer.
 * * -AML_ENOMEM if their was not enough memory on host or device to
 *fulfill the request.
 */
void *aml_area_opencl_mmap(const struct aml_area_data *area_data,
                           size_t size,
                           struct aml_area_mmap_options *options);

/**
 * munmap hook for aml area created with `aml_area_opencl_create()`.
 * @param[in] area_data: Area data of type `struct aml_area_opencl_data`
 * where flag field contains `cl_mem_flags`.
 * @param[in] ptr: A pointer created with `aml_area_opencl_mmap()` and same type
 * of area_data.
 * @param[in] size: Allocation size.
 * @return AML_SUCCESS on success.
 * @return -AML_EINVAL if ptr is not a valid `cl_mem` buffer.
 * @return -AML_ENOMEM if their was not enough memory on host or device to
 *fulfill the request.
 */
int aml_area_opencl_munmap(const struct aml_area_data *area_data,
                           void *ptr,
                           const size_t size);

/** aml OpenCL SVM area hooks. **/
extern struct aml_area_ops aml_area_opencl_svm_ops;

/**
 * This will spawn areas mapping shared virtual memory, i.e `aml_area_mmap()` on
 * these areas will create a pointer usable both on host and device.
 * @see
 * https://www.khronos.org/registry/OpenCL/specs/2.2/html/OpenCL_API.html#shared-virtual-memory
 * This area creator does not allow mapping of memory regions where OpenCL is
 * responsible for synchronization with other APIs.
 * @see
 * https://www.khronos.org/registry/OpenCL/specs/2.2/html/OpenCL_API.html#context-properties-table
 * @param[out] area: A pointer to the area to allocate.
 * @param[in] context: A valid OpenCL context.
 * @param[in] flags: Flags to pass to clSVMalloc().
 * @param[in] alignement: The minimum alignment in bytes that is required for
 * the newly created buffer’s memory region.
 * @see
 * https://www.khronos.org/registry/OpenCL//sdk/2.1/docs/man/xhtml/clSVMAlloc.html
 * @return AML_SUCCESS on success.
 * @return -AML_ENOMEM if there were not enough memory available to allocate
 * area.
 */
int aml_area_opencl_svm_create(struct aml_area **area,
                               cl_context context,
                               const cl_svm_mem_flags flags,
                               cl_uint alignement);

/**
 * OpenCL implementation of mmap operation for aml area created with
 * `aml_area_opencl_svm_create()`.
 * @param[in] area_data: Area data of type `struct aml_area_opencl_data`
 * where flag field contains `struct aml_area_opencl_svm_flags`.
 * @param[in] size: The size to allocate.
 * @param[in] options: unused.
 * @see aml_area
 * @return A valid SVM pointer on success.
 * @return NULL on error. This maybe the result of unsupported flags in
 * area.
 * @see
 * https://www.khronos.org/registry/OpenCL//sdk/2.1/docs/man/xhtml/clSVMAlloc.html
 */
void *aml_area_opencl_svm_mmap(const struct aml_area_data *area_data,
                               size_t size,
                               struct aml_area_mmap_options *options);

/**
 * munmap hook for aml area created with `aml_area_opencl_svm_create()`.
 * @param[in] area_data: Area data of type `struct aml_area_opencl_data`
 * where flag field contains `struct aml_area_opencl_data`.
 * @param[in] ptr: A pointer created with `aml_area_opencl_svm_mmap()` and same
 * type of area_data.
 * @param[in] size: unused.
 * @return AML_SUCCESS
 */
int aml_area_opencl_svm_munmap(const struct aml_area_data *area_data,
                               void *ptr,
                               const size_t size);

/**
 * \brief OpenCL area destruction.
 *
 * Destroy (finalize and free resources) a struct aml_area created by
 * aml_area_opencl_create() or aml_area_opencl_svm_create().
 *
 * @param[in, out] area is NULL after this call.
 **/
void aml_area_opencl_destroy(struct aml_area **area);

/**
 * @}
 **/

#endif // AML_AREA_OPENCL_H
