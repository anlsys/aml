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

/* when the allocator is out of mapped memory, it mmaps chunks of this size */
# define AML_ALLOCATOR_MMAP_SIZE (512*1024*1024)

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

/** Allocator required methods. See structure definition */
struct aml_allocator_ops;

/** User level allocator structure. */
struct aml_allocator {
    /** The area to map new regions */
    struct aml_area *area;
    /** The area options */
    struct aml_area_mmap_options *opts;
	/** methods */
	struct aml_allocator_ops *ops;
	/** Allocator lock **/
	pthread_mutex_t lock;
};

/** Allocator internal's chunk information */
struct aml_allocator_chunk {
	/** memory allocator for the user (read-only) */
	void *ptr;
	/** size of the chunk, greater or equals to the size requested by the
	 * user (read-only) */
	size_t size;
	/** an opaque object that the user can attach to the chunk (read/write)	 */
	void *user_data;
};

# define AML_ALLOCATOR_CREATE_FAIL(ALLOCATOR_ADDR, AREA, OPTS, OPS, TYPE, NAME)     \
    do {                                                                            \
        free(NAME);                                                                 \
        return -AML_FAILURE;                                                        \
    } while (0)

# define AML_ALLOCATOR_CREATE_BEGIN(ALLOCATOR_ADDR, AREA, OPTS, OPS, TYPE, NAME)    \
    do {                                                                            \
        if (ALLOCATOR_ADDR == NULL || AREA == NULL)                                 \
            return -AML_EINVAL;                                                     \
        TYPE * NAME = (TYPE *) calloc(1, sizeof(TYPE));                             \
        if (NAME == NULL)                                                           \
            return -AML_ENOMEM;                                                     \
        if (pthread_mutex_init(&NAME->super.lock, NULL) != 0)                       \
            AML_ALLOCATOR_CREATE_FAIL(ALLOCATOR_ADDR, AREA, OPTS, OPS, TYPE, NAME); \
        NAME->super.area = AREA;                                                    \
        NAME->super.opts = OPTS;                                                    \
        NAME->super.ops = OPS;

# define AML_ALLOCATOR_CREATE_END(ALLOCATOR_ADDR, AREA, OPTS, OPS, TYPE, NAME)      \
        *ALLOCATOR_ADDR = &(NAME->super);                                           \
        return AML_SUCCESS;                                                         \
    } while (0)

# define AML_ALLOCATOR_DESTROY_BEGIN(ALLOCATOR_ADDR, TYPE, NAME)                    \
    do {                                                                            \
        if (ALLOCATOR_ADDR == NULL || (*ALLOCATOR_ADDR) == NULL)                    \
            return -AML_EINVAL;                                                     \
        TYPE * NAME = (TYPE *) (*ALLOCATOR_ADDR);                                   \
        pthread_mutex_lock(&NAME->super.lock);

# define AML_ALLOCATOR_DESTROY_FAIL(ALLOCATOR_ADDR, TYPE, NAME)                     \
    do {                                                                            \
        pthread_mutex_unlock(&NAME->super.lock);                                    \
        return -AML_FAILURE;                                                        \
    } while (0)

# define AML_ALLOCATOR_DESTROY_END(ALLOCATOR_ADDR, TYPE, NAME)                      \
        pthread_mutex_unlock(&NAME->super.lock);                                    \
        if (pthread_mutex_destroy(&NAME->super.lock) != 0)                          \
            return -AML_FAILURE;                                                    \
        free(*ALLOCATOR_ADDR);                                                      \
        *ALLOCATOR_ADDR = NULL;                                                     \
        return AML_SUCCESS;                                                         \
    } while (0)

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
	void *(*alloc)(struct aml_allocator *alloc, size_t size);

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
	int (*free)(struct aml_allocator *alloc, void *ptr);

    /**
     * Optional method.
     * @see aml_allocator_give()
     */
    int (*give)(struct aml_allocator *alloc, void * ptr, size_t size);

	/**
	 *  Optional method.
	 *  @see aml_allocator_alloc_chunk()
	 */
	struct aml_allocator_chunk *(*alloc_chunk)(
	        struct aml_allocator *alloc, size_t size);

	/**
	 *  Optional method.
	 *  @see aml_allocator_free_chunk()
	 */
	int (*free_chunk)(struct aml_allocator *alloc,
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
 *  Give a chunk of memory to the allocator, so it can use it to allocate future allocations.
 *  The chunk will be unmap by the allocator.
 *  @param[in,out] allocator: The allocator which is given memory
 *  @param[in] chunk: the chunk info that will be copied internally by the allocator\
 *  @return AML_SUCCESS on success or an appropriate aml error code.
 */
int aml_allocator_give(struct aml_allocator *allocator, void * ptr, size_t size);

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
