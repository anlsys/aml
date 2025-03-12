/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#ifndef __AML_HIGHER_ALLOCATOR_H_
#define __AML_HIGHER_ALLOCATOR_H_

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @}
 * @defgroup aml_allocator "AML Allocator"
 * @brief Data Allocator.
 *
 * Allocator is a building block on top of `areas` to optimize the process
 * of frequently requesting and freeing memory.
 *
 * @see aml_area
 * @{
 **/

/** User defined allocator metadata */
struct aml_allocator_data;
/** Allocator required methods. See structure definition */
struct aml_allocator_ops;

/** User level allocator structure. */
struct aml_allocator {
	/** metadata */
	struct aml_allocator_data *data;
	/** methods */
	struct aml_allocator_ops *ops;
};

/** Allocator internal's chunk information */
struct aml_allocator_chunk {
	/** memory allocator for the user (read-only) */
	void *ptr;
	/** size of the chunk, greater or equals to the size requested by the
	 * user (read-only) */
	size_t size;
	/** an opaque object that the user can attach to the chunk (read/write)
	 */
	void *user_data;
};

/**
 * Allocator methods.
 * The design pattern of aml allocator is design to meet simplicity and
 * expected interface.
 *
 * Interface Motivation:
 *
 * Most device backends do not require to match device pointers with any
 * associated metadata (e.g size):
 * - Cuda:
 * `cudaError_t cudaFree ( void* devPtr )`
 * @see
<https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1ga042655cbbf3408f01061652a075e094>
 * - Level Zero:
 * `ze_result_t zeMemFree(ze_context_handle_t hContext, void *ptr)`
 * Context handle in \ref aml_area_ze are stored in the area.
 * @see <https://spec.oneapi.com/level-zero/latest/core/api.html#zememfree>
 * - OpenCL:
 * `cl_int clReleaseMemObject(cl_mem memobj)`
 * @see <http://man.opencl.org/clReleaseMemObject.html>
 * These interfaces show that backends manage the association
 * pointer/metadata internally. However, they are not always optimized for the
 * use case of frequent allocation/free. Implementing an allocator for such
 * a use case may require to implement this association in order to manage a
 * memory pool.
 *
 * Storing metadata about device pointers would require either
 * device to host transfers to:
 * - read this metadata embedded in the pointer on allocation/free or,
 * - hashtable reads and write to retrieve/create the metadata associated with
 * a device pointer or,
 * - provide the user with the metadata/pointer association.
 *
 * The first solution appears to be complex to implement and poor
 * performance wise. The last solution appears to break expected interface and
 * simplicity. In most cases, it is possible to implement specific purpose
 * allocators that will allow to stick to a simple interface while keeping
 * allocation/free routine performant, i.e avoiding device to host transfers,
 * or search structure lookups.
 */
struct aml_allocator_ops {
	/**
	 * Required method.
	 * Normal allocation routine.
	 *
	 * @param[in, out] data: The allocator metadata.
	 * @param[in] size: The minimum allocation size.
	 * @return NULL on error with aml_errno set to the appropriate error
	 * code.
	 * @return A pointer to the beginning of the allocation on success.
	 */
	void *(*alloc)(struct aml_allocator_data *data, size_t size);

	/**
	 * Required method.
	 * Release memory associated with a pointer obtained with this
	 * allocator.
	 *
	 * @param[in, out] data: The allocator metadata.
	 * @param[in, out] ptr: The pointer allocated with the same allocator
	 * to free.
	 * @return AML_SUCCESS on success or an appropriate aml error code.
	 */
	int (*free)(struct aml_allocator_data *data, void *ptr);

	/**
	 *  Optional method.
	 *  @see aml_allocator_alloc_chunk()
	 */
	struct aml_allocator_chunk *(*alloc_chunk)(
	        struct aml_allocator_data *data, size_t size);

	/**
	 *  Optional method.
	 *  @see aml_allocator_free_chunk()
	 */
	int (*free_chunk)(struct aml_allocator_data *data,
	                  struct aml_allocator_chunk *chunk);
};

/**
 * Allocate memory with an allocator.
 *
 * @param[in, out] allocator: The allocator to use.
 * @param[in] size: The minimum allocation size.
 * @return NULL on error with aml_errno set to the appropriate error
 * code.
 * @return A pointer to the beginning of the allocation on success.
 */
void *aml_alloc(struct aml_allocator *allocator, size_t size);

/**
 * Release memory associated with a pointer obtained from a call to
 * aml_alloc().
 *
 * @param[in, out] allocator: The allocator used to allocate pointer.
 * @param[in, out] ptr: The pointer allocated with the same allocator.
 * @return AML_SUCCESS on success or an appropriate aml error code.
 */
int aml_free(struct aml_allocator *allocator, void *ptr);

/**
 * Allocate memory with an allocator.
 *
 * @param[in, out] allocator: The allocator to use.
 * @param[in] size: The minimum allocation size.
 * @return NULL on error with aml_errno set to the appropriate error
 * code.
 * @return The chunk of memory allocated.
 */
struct aml_allocator_chunk *
aml_allocator_alloc_chunk(struct aml_allocator *allocator, size_t size);

/**
 * Release memory associated with the chunk obtained from a call to
 * aml_allocator_alloc_chunk().
 *
 * @param[in, out] allocator: The allocator used to allocate pointer.
 * @param[in, out] ptr: The chunk allocated with the same allocator.
 * @return AML_SUCCESS on success or an appropriate aml error code.
 */
int aml_allocator_free_chunk(struct aml_allocator *allocator,
                             struct aml_allocator_chunk *chunk);

/**
 * @}
 **/

#ifdef __cplusplus
}
#endif

#endif // __AML_HIGHER_ALLOCATOR_H_
