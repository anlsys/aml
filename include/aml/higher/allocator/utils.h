/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#ifndef __AML_HIGHER_ALLOCATOR_UTILS_H_
#define __AML_HIGHER_ALLOCATOR_UTILS_H_

#ifdef __cplusplus
extern "C" {
#endif

/** One contiguous chunk of memory */
struct aml_memory_chunk {
	// The pointer to memory starting the chunk.
	void *ptr;
	// Chunk size.
	size_t size;
};

/**
 * Comparison function betwenn two chunks.
 * This function can be used to sort an array of chunks.
 * The comparison is a comparison between chunks pointer values.
 *
 * @param[in] lhs: A pointer to a `struct aml_memory_chunk`, left
 * hand side of the comparison operator.
 * @param[in] rhs: A pointer to a `struct aml_memory_chunk`, right
 * hand side of the comparison operator.
 * @return 1 if rhs > lhs
 * @return 0 if rhs = lhs
 * @return -1 if rhs < lhs
 */
int aml_memory_chunk_comp(const void *lhs, const void *rhs);

/**
 * Check if two chunks match, that is, if the next
 * byte of one of the chunks is the memory pointer of the other chunk.
 * When two chunks match, they can be easily merged into one.
 *
 * @param[in] a: A chunk that can be either the left hand side or the
 * right hand side of the match.
 * @param[in] b: A chunk that can be either the left hand side or the
 * right hand side of the match.
 * @return 1 if chunks match else 0.
 */
int aml_memory_chunk_match(const struct aml_memory_chunk a,
                           const struct aml_memory_chunk b);

/**
 * Check if two chunks overlap.
 *
 * @param[in] a: A chunk.
 * @param[in] b: Another chunk.
 * @return 1 if chunks overlap else 0.
 */
int aml_memory_chunk_overlap(const struct aml_memory_chunk a,
                             const struct aml_memory_chunk b);

/**
 * Check if a chunk contain another chunk. Limits (start and end) may
 * be the same.
 *
 * @param[in] super: The chunk that may contain the other.
 * @param[in] b: The chunk that may be contained in the other.
 *
 * @return 1 if `super` contains `sub` else 0.
 */
int aml_memory_chunk_contains(const struct aml_memory_chunk super,
                              const struct aml_memory_chunk sub);

/** One pool of memory. */
struct aml_memory_pool {
	// The chunk of memory that this pool manages.
	struct aml_memory_chunk memory;
	// The chunks of memory from the global `memory` chunk that
	// still available for allocation stored in a utarray.
	void *chunks;
};

/**
 * Create a new pool of memory.
 * This function does not allocate the actual underlying pointer to user
 * memory.
 *
 * @param[out] out: A pointer where the new pool of memory is allocated.
 * @param[in] ptr: A pointer to the total memory managed by this pool.
 * The pointer must point to a valid chunk of memory to manage.
 * @param[in] size: The size of total memory managed by this pool.
 *
 * @return AML_SUCCESS on success.
 * @return -AML_ENOMEM if there was not enough memory to satisfy the call.
 * @return -AML_EINVAL if `out` is NULL or `size` is 0.
 */
int aml_memory_pool_create(struct aml_memory_pool **out,
                           void *ptr,
                           const size_t size);

/**
 * Destroy a pool of memory.
 * This function does not release the underlying user memory.
 *
 * @param[out] out: A pointer where the new pool of memory is allocated.
 * @param[out] memory: The total memory managed by this pool.
 *
 * @return AML_SUCCESS on success.
 * @return -AML_ENOMEM if there was not enough memory to satisfy the call.
 */
int aml_memory_pool_destroy(struct aml_memory_pool **out);

/**
 * Pop a chunk of memory out of the pool.
 *
 * @param[in] pool: An initialized memory pool.
 * @param[out] out: Where to store the returned pointer.
 * @param[in] size: The size of the chunk to pop from the pool.
 *
 * @return AML_SUCCESS on success.
 * @return -AML_ENOMEM if there was not enough memory left in the pool
 * to satisfy the call. In that case, `out` is not updated.
 * @return -AML_EINVAL if `pool` is `NULL` or `out` is `NULL` or `size` is 0.
 */
int aml_memory_pool_pop(struct aml_memory_pool *pool,
                        void **out,
                        const size_t size);

/**
 * Insert a chunk of memory in the pool.
 * The total pool managed memory must be able to contain the chunk to
 * insert. The chunk to insert must not overlap with any other chunk
 * in the pool.
 *
 * @param[in] pool: An initialized memory pool.
 * @param[in] ptr: The pointer to memory to insert in the pool.
 * @param[in] size: The size of the chunk to insert the pool.
 *
 * @return AML_SUCCESS on success.
 * @return -AML_EDOM if the chunk to insert crosses the pool boundary.
 * @return -AML_EINVAL if the chunk to insert overlap with another chunk in
 * the pool.
 * @return -AML_ENOMEM if there was not enough computer memory to satisfy
 * the call.
 */
int aml_memory_pool_push(struct aml_memory_pool *pool,
                         void *ptr,
                         const size_t size);

/**
 * @}
 **/

#ifdef __cplusplus
}
#endif

#endif // __AML_HIGHER_ALLOCATOR_UTILS_H_
