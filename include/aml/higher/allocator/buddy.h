/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#ifndef __AML_HIGHER_ALLOCATOR_BUDDY_H_
#define __AML_HIGHER_ALLOCATOR_BUDDY_H_

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @}
 * @defgroup aml_allocator_buddy "AML Allocator Buddy"
 * @brief Allocator for constant size elements.
 *
 * This allocator create memory pools and returns constant buddy chunks of the
 * pool.
 *
 * @{
 **/

/**
 * The low-level structure managing chunks of memory.
 * This structure however does not manage memory mapping or concurrency.
 */
struct buddy_allocator;

/**
 * Allocate a low-level buddy allocator.
 *
 * @param[out] out: The pointer where to allocate the allocator.
 * `out` must not be NULL.
 *
 * @return -AML_ENOMEM if there was not enough memory to allocate the
 * allocator.
 */
int buddy_allocator_create(struct buddy_allocator **out);

/**
 * Add mapped memory to a low-level buddy allocator.
 * You must not add a memory range already in use by the allocator.
 *
 * @param[in] b: The allocator where to add mapped memory. `b` must not be NULL.
 * @param[in] ptr: The pointer to the beginning of the mapped memory region.
 * @param[in] size: The size of the mapped memory region. `size` must be
 * greater than 0 and a power of two.
 *
 * @return AML_SUCCESS on success.
 * @return -AML_ENOMEM if there was not enough memory to extend allocator.
 * @return -AML_EINVAL if `ptr` is NULL or size is 0 or not a power of two.
 */
int buddy_allocator_extend(struct buddy_allocator *b, void *ptr, size_t size);

/**
 * Delete a low-level buddy allocator.
 * This does not unmap associated memory. The function fails if not all
 * memory associated with this allocator has been freed.
 * This function does not check its input. See argument for assumed values.
 *
 * @param[in, out] buddy: The allocator to delete. buddy is set to NULL
 * on success. `buddy` must not be NULL.
 *
 * @return -AML_EBUSY If not all memory used by allocator has been freed.
 * @return AML_SUCCESS otherwise.
 */
int buddy_allocator_destroy(struct buddy_allocator **buddy);

/**
 * Free memory allocated with this low-level buddy allocator.
 * This function does not check its input. See argument for assumed values.
 *
 * @param[in] b: The allocator that yielded ptr. `b` must not be NULL.
 * @param[in] ptr: The pointer to free.
 *
 * @return -AML_EINVAL if `ptr` is not a pointer that has been allocated
 * by `buddy` or if `ptr` has already been freed.
 * @return AML_SUCCESS on success.
 */
int buddy_allocator_free(struct buddy_allocator *b, void *ptr);

/**
 * Allocate memory with a low-level buddy allocator.
 * This function does not check its input. See argument for assumed values.
 *
 * @param[in] b: An initialized buddy allocator. `b` must not be NULL.
 * @param[out] out: Where to store the pointer to new allocation. `out`
 * must not be NULL.
 * @param[in] size: The minimum size of the allocation.
 * `size` must not be 0.
 *
 * @return -AML_ENOMEM if allocator can't satisfy this allocation because
 * either the allocator is out of memory or the system is out of memory.
 * @return AML_SUCCESS on success.
 */
int buddy_allocator_alloc(struct buddy_allocator *b, void **out, size_t size);

/** User buddy allocator structure. */
struct aml_allocator_buddy {
    /** The mapped memory regions stored in a utarray. */
    void *pools;
    /** The structure managing memory chunks. */
	struct buddy_allocator *allocator;
	/** Allocator lock **/
	pthread_mutex_t lock;
	/** The area to map new regions */
	struct aml_area *area;
	/** The area options */
	struct aml_area_mmap_options *opts;
};

/** Buddy allocator methods. */
extern struct aml_allocator_ops aml_allocator_buddy_ops;

/**
 * User level buddy allocator constructor.
 *
 * @param[out] allocator: A pointer to where to store the newly created
 * allocator.
 * @param[in] area: The area used to map memory. Area must yield pointers
 * on which pointer arithmetic is a valid operation within mapping bounds.
 * @param[in] opts: The area options.
 *
 * @return AML_SUCCESS on success.
 * @return -AML_EINVAL if `allocator` is NULL, or `area` is NULL.
 * @return -AML_ENOMEM There was not enough memory to create an allocator.
 */
int aml_allocator_buddy_create(struct aml_allocator **allocator,
                               struct aml_area *area,
                               struct aml_area_mmap_options *opts);

/**
 * Buddy allocator destructor.
 * Upon successful destruction, all data allocated with this allocator is
 * also unmapped.
 *
 * @param[in, out] allocator: The allocator to destroy. `allocator` is set to
 * NULL on success.
 *
 * @return -AML_EBUSY If not all memory used by allocator has been freed.
 * @return AML_SUCCESS otherwise.
 * @return Others aml error codes from aml_area_munmap() result if this
 * function fails. In this case, the allocator is left in an inconsistent
 * state and cannot be used for any other reason than destroying it.
 */
int aml_allocator_buddy_destroy(struct aml_allocator **allocator);

/**
 * Buddy allocator `alloc()` method.
 *
 * @param[in] data: The user level allocator.
 * @param[in] size: The minimum allocation size.
 *
 * @return A pointer to the beginning of the allocation on success.
 * @return NULL on error with aml_errno set to the appropriate error.
 * aml_errno can be one of the following:
 * + -AML_ENOMEM if allocator can't satisfy this allocation because
 * either the allocator is out of memory, the system is out of memory,
 * or the target area cannot map more memory.
 * + An other aml error code from `aml_area_mmap()`.
 */
void *aml_allocator_buddy_alloc(struct aml_allocator_data *data, size_t size);

/**
 * Buddy allocator `free()` method.
 *
 * @param[in] data: The user level allocator with which `ptr` was obtained.
 * @param[in] ptr: The pointer to free.
 *
 * @return AML_SUCCESS on success and pointer is freed.
 * @return -AML_EINVAL if `ptr` is not a pointer that has been allocated
 * by this allocator or if `ptr` has already been freed.
 */
int aml_allocator_buddy_free(struct aml_allocator_data *data, void *ptr);

/**
 * @}
 **/

#ifdef __cplusplus
}
#endif

#endif // __AML_HIGHER_ALLOCATOR_BUDDY_H_
