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

#include "aml/higher/allocator.h"

#include "../benchmarks/utils.h"
#include "memset.h"

// Struct definition below
struct aml_alloc_record;

//------------------------------------------------------------------------------
// `size_fn` options
//------------------------------------------------------------------------------

/**
 * Function computing what should be the next allocation size.
 * This function calls the underlying implementation to get the next
 * allocation size.
 * If the function succeed, the record counter is incremented after
 * the underlying implementation call.
 * @param[in, out] record: The ongoing record for allocation.
 * @param[out] size: The output size for the next allocation.
 * @return -AML_EINVAL if the `record` or `size` is `NULL`.
 * @return the output of the underlying implementation. In practice,
 * this is always AML_SUCCESS.
 */
int aml_alloc_record_next_size(struct aml_alloc_record *record, size_t *size);

/** Function pointer of the underlying implementation of
 * `aml_alloc_record_next_size()`. */
typedef int (*aml_alloc_record_next_size_fn)(
        const struct aml_alloc_record *record, size_t *size);

/** `aml_alloc_record_next_size_fn` returning a random size. */
int rand_size(const struct aml_alloc_record *record, size_t *size);
/** `aml_alloc_record_next_size_fn` returning record base size. */
int base_size(const struct aml_alloc_record *record, size_t *size);
/** `aml_alloc_record_next_size_fn` returning record max_size. */
int max_size(const struct aml_alloc_record *record, size_t *size);
/** `aml_alloc_record_next_size_fn` returning a size increased by base on
 * each iteration and cycling when max_size is reached. */
int increasing_size(const struct aml_alloc_record *record, size_t *size);

//------------------------------------------------------------------------------
// `next_fn` options
//------------------------------------------------------------------------------

/**
 * This function returns the next index in the record array of pointers.
 * The index may point to an empty slot, in which case the slot can only
 * be used for allocation.
 * The function may not return any index on purpose (skipping a step),
 * in which case `-AML_EDOM` is returned.
 * This function main purpose is to select a slot to free in the
 * allocation/free test or benchmark scheme.
 * @param[in] record: The record used to select a slot.
 * @param[out] index: The output where to store slot index.
 * @return -AML_EINVAL if the `record` or `size` is `NULL`.
 * @return -AML_EDOM if the slot cannot be used.
 * @return The underlying implementation return value otherwise.
 */
int aml_alloc_record_next_free(const struct aml_alloc_record *record,
                               size_t *index);

/** Function pointer of the underlying implementation of
 * `aml_alloc_record_next_free()`. */
typedef int (*aml_alloc_record_next_free_fn)(const struct aml_alloc_record *,
                                             size_t *);

/** `next_free` not chosing a slot. */
int no_pick(const struct aml_alloc_record *record, size_t *index);
/** `next_free` picking previous allocation. */
int pick_prev(const struct aml_alloc_record *record, size_t *index);
/** `next_free` picking random allocation or nothing. */
int pick_rand(const struct aml_alloc_record *record, size_t *index);

//------------------------------------------------------------------------------
// Generic benchmark / test
//------------------------------------------------------------------------------

// Structure to keep track of on going allocated pointers and benchmark
// performance.
struct aml_alloc_record {
	// Allocated pointers in a utarray.
	void *ptrs;
	// Maximum allocation size
	size_t max_size;
	// Allocation base size. Allocations must be a multiple of this number.
	size_t base_size;
	// Counter of allocations.
	size_t counter;
	// Delay before freeing any allocations.
	size_t delay;
	// Function returning the size of the next allocation.
	aml_alloc_record_next_size_fn next_size;
	// Function returning the slot of the next free.
	aml_alloc_record_next_free_fn next_free;
};

struct aml_alloc_workflow_output {
	size_t total_size;
	size_t allocated_size;
	size_t max_allocated_size;
	size_t num_alloc;
	size_t num_free;
	size_t last_size;
	size_t num_iterations;
};

void aml_alloc_workflow_output_print(FILE *out,
                                     struct aml_alloc_workflow_output);

/**
 * This is a configurable test function.
 * This function runs a test case as follow:
 * + Create a vector of pointers.
 * + For `num_iterations`,
 *   - allocate a pointer of size specified by `next_size`. Allocation
 * time is measured and accumulated in `alloc_stats`.
 *   - Append pointer to the vector of pointers.
 *   - Chose a spot in the vector of pointer with `next_free`.
 * If the function does not return -AML_EDOM, then free the pointer
 * in that spot and remove it from the vector. Free time is measured and
 * accumulated in `free_stats`.
 *
 * @param[in] max_size: The maximum allocation size.
 * @param[in] base_size: Every allocation is a multiple of this size.
 * `base_size` must be greater than 0.
 * @param[in] delay: The number of allocations before starting to eventually
 * free some allocations.
 * @param[in] next_size: A function that returns the size for the next
 * allocation.
 * @param[in] next_free: A function that returns an index in the array
 * of on flight allocations. The allocation matching this index will be
 * freed.
 * @param[in, out] num_iterations: The number of allocations to perform.
 * On exit, `num_iterations` contains the number of remaining allocations.
 * If the function succeeds all the way, `num_iterations` should be 0 on
 * exit.
 * @param[in] allocator: The allocator to use for allocations and frees.
 * @param[out] alloc_stats: Where to store the average, max and min
 * allocation times.
 * @param[out] free_stats: Where to store the average, max and min
 * free times.
 *
 * @return AML_SUCCESS on success.
 * @return -AML_EINVAL if `base_size` is 0, or `next_size` is NULL, or
 * `next_free` is NULL, or `num_iterations` is NULL, or `allocator` is NULL.
 * @return The value stored in `-aml_errno` if an allocation fails.
 *
 * @see `aml_alloc_record_next_free()`
 * @see `aml_alloc_record_next_size()`
 */
int aml_alloc_workflow_run(const size_t max_size,
                           const size_t base_size,
                           const size_t delay,
                           aml_alloc_record_next_size_fn next_size,
                           aml_alloc_record_next_free_fn next_free,
                           const size_t num_iterations,
                           struct aml_allocator *allocator,
                           aml_memset_fn memset_fn,
                           struct aml_time_stats *alloc_stats,
                           struct aml_time_stats *free_stats,
                           struct aml_alloc_workflow_output *out);
