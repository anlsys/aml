/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#ifndef __AML_HIGHER_ALLOCATOR_FREELIST_H_
#define __AML_HIGHER_ALLOCATOR_FREELIST_H_

#ifdef __cplusplus
extern "C" {
#endif

struct freelist_allocator;

typedef enum    aml_allocator_freelist_chunk_state_t
{
    FREELIST_CHUNK_STATE_FREE       = 0,
    FREELIST_CHUNK_STATE_ALLOCATED  = 1,

}               aml_allocator_freelist_chunk_state_t;

/**
 * Represent a segment of memory in device memory (used by custom allocator)
 * It is placed in two chained list:
 *  - the list of all chunk in device memory
 *  - the list of free chunk in device memory
*/
struct  aml_allocator_freelist_chunk
{
    struct aml_allocator_chunk super;
    aml_allocator_freelist_chunk_state_t state;       /* state of the chunk */
    struct aml_allocator_freelist_chunk * prev;       /* previous chunk in double chained list */
    struct aml_allocator_freelist_chunk * next;       /* next chunk in double chained list */
    struct aml_allocator_freelist_chunk * freelink;   /* next freechunk in the chained list */
};

struct aml_allocator_freelist
{
	/** The area to map new regions */
	struct aml_area *area;

	/** The area options */
	struct aml_area_mmap_options *opts;

	/** Allocator lock **/
	pthread_mutex_t lock;

    /* the free list */
    struct aml_allocator_freelist_chunk * free_chunk_list;
};

/** Freelist allocator methods. */
extern struct aml_allocator_ops aml_allocator_freelist_ops;

/**
 * User level freelist allocator constructor.
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
int aml_allocator_freelist_create(struct aml_allocator **allocator,
                               struct aml_area *area,
                               struct aml_area_mmap_options *opts);

/**
 * Freelist allocator destructor.
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
int aml_allocator_freelist_destroy(struct aml_allocator **allocator);

#ifdef __cplusplus
}
#endif

#endif // __AML_HIGHER_ALLOCATOR_FREELIST_H_
