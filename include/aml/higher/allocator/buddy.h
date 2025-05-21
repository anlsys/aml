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

struct aml_allocator_buddy;

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
void *aml_allocator_buddy_alloc(struct aml_allocator *alloc, size_t size);

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
int aml_allocator_buddy_free(struct aml_allocator *alloc, void *ptr);

/**
 * @}
 **/

#ifdef __cplusplus
}
#endif

#endif // __AML_HIGHER_ALLOCATOR_BUDDY_H_
