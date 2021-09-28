/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#ifndef __AML_ALLOCATOR_AREA_H_
#define __AML_ALLOCATOR_AREA_H_

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @}
 * @defgroup aml_area "AML Area Allocator"
 * @brief Bearbone allocator based on areas backend.
 *
 * This allocator uses AML area abstraction for allocations and frees.
 * Areas of this allocator are assumed to use no options and to be
 * able to unmap memory without passing the size argument.
 *
 * @{
 **/

/**
 * Area allocator data.
 * Area is not owned by the allocator, therefore, it has
 * to live longer than the allocator.
 */
struct aml_allocator_area_data {
	struct aml_area *area;
	struct aml_area_mmap_options *opts;
	pthread_mutex_t lock;
	void *chunks;
};

/** Area allocator operation table. */
extern struct aml_allocator_ops aml_allocator_area_ops;

/**
 * Create an area allocator from an area.
 * Ownership of the area is not acquired, therefore,
 * the area has to live longer than the allocator.
 */
int aml_allocator_area_create(struct aml_allocator **out,
                              struct aml_area *area,
                              struct aml_area_mmap_options *opts);

/** Destroy an area allocator. The embedded area is not destroyed. */
int aml_allocator_area_destroy(struct aml_allocator **allocator);

/**
 * `alloc()` method of the allocator.
 * Passes `NULL` to `aml_area_mmap()` options argument.
 */
void *aml_allocator_area_alloc(struct aml_allocator_data *data, size_t size);

/**
 * `free()` method of the allocator.
 * Passes `0` to `area_munmap()` size argument.
 */
int aml_allocator_area_free(struct aml_allocator_data *data, void *ptr);

/**
 * @}
 **/

#ifdef __cplusplus
}
#endif

#endif // __AML_ALLOCATOR_AREA_H_
