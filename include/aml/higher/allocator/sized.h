/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#ifndef __AML_HIGHER_ALLOCATOR_SIZED_H_
#define __AML_HIGHER_ALLOCATOR_SIZED_H_

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @}
 * @defgroup aml_allocator_sized "AML Allocator Sized"
 * @brief Allocator for constant size elements.
 *
 * This allocator create memory pools and returns constant sized chunks of the
 * pool.
 *
 * @{
 **/

/** Sized allocator metadata. */
struct aml_allocator_sized {
	/** The size of user allocations. */
	size_t chunk_size;
	/** The free mapped memory regions (internally a utlist) */
	void *free_pools;
	/** The free mapped memory regions (internally a uthash) */
	void *occupied_pools;
	/** The area to map new regions */
	struct aml_area *area;
	/** The area options */
	struct aml_area_mmap_options *opts;
};

/** Sized allocator methods. */
extern struct aml_allocator_ops aml_allocator_sized;

/**
 * Sized allocator constructor.
 *
 * @param[out] allocator: A pointer to where to store the newly created
 * allocator.
 * @param[in] size: The size of each allocation. The user requested size
 * using `aml_allocator_sized_alloc()` shall not exceed this size.
 * @param[in] area: The area used to map memory. Area must yield pointers
 * on which pointer arithmetic is a valid operation within mapping bounds.
 * @param[in] opts: The area options.
 *
 * @return AML_SUCCESS on success.
 * @return -AML_EINVAL if allocator is NULL, size is 0 or area is NULL.
 * @return -AML_ENOMEM There was not enough memory to create an allocator or
 * associated memory pool.
 */
int aml_allocator_sized_create(struct aml_allocator **allocator,
                               size_t size,
                               struct aml_area *area,
                               struct aml_area_mmap_options *opts);

/**
 * Sized allocator destructor.
 * Upon successful destruction, all data allocated with this allocator is also
 * unmapped.
 *
 * @param[in, out] allocator: The allocator to destroy. `allocator` is set to
 * NULL on success.
 *
 * @return AML_SUCCESS.
 */
int aml_allocator_sized_destroy(struct aml_allocator **allocator);

/**
 * Sized allocator `alloc()` method.
 * Allocations are aligned on allocator size.
 * If size is greater than set allocator size, then NULL is returned and
 * aml_errno is set to -AML_EINVAL.
 *
 * @param[in, out] data: The allocator metadata (struct aml_allocator_sized *).
 * @param[in] size: The minimum allocation size. Must be less than or equal to
 * allocator allocations size.
 *
 * @return A pointer to the beginning of the allocation on success.
 * @return NULL on error with aml_errno set to the appropriate error.
 */
void *aml_allocator_sized_alloc(struct aml_allocator_data *data, size_t size);

/**
 * Sized allocator `free()` method.
 *
 * @param[in, out] data: The allocator metadata (struct aml_allocator_sized *).
 * @param[in, out] ptr: The pointer allocated with the same allocator
 * to free.
 *
 * @return AML_SUCCESS on success and pointer is freed.
 * @return -AML_EINVAL if pointer is not a valid with this allocator and
 * pointer is not freed.
 */
int aml_allocator_sized_free(struct aml_allocator_data *data, void *ptr);

/**
 * @}
 **/

#ifdef __cplusplus
}
#endif

#endif // __AML_HIGHER_ALLOCATOR_SIZED_H_
