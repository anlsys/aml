/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include "test_allocator.h"

#include "aml/higher/allocator/sized.h"

#include "dummy_area.h"

int main(int argc, char **argv)
{
	assert(aml_init(&argc, &argv) == AML_SUCCESS);

	int err;
	const size_t base_s = 1UL << 6; // 64B
	const size_t max_s = 1UL << 12; // 4096B
	// const size_t total_s = 1UL << 20; // 1MiB
	struct aml_allocator *allocator;
	size_t num_iterations;

	assert(aml_allocator_sized_create(&allocator, base_s, &aml_area_dummy,
	                                  NULL) == AML_SUCCESS);

	// Alloc consecutive base size and free previous alloc.
	// Total allocated size is greater than total pool size.
	// However at every step, only base_s is out of the memory pool.
	// This test should not fail.
	num_iterations = 1000;
	err = aml_alloc_workflow_run(max_s, base_s, 0, base_size, pick_prev,
	                             num_iterations, allocator, NULL, NULL,
	                             NULL);
	assert(err == AML_SUCCESS);

	// Alloc random sizes and free random alloc.
	num_iterations = 1000;
	err = aml_alloc_workflow_run(max_s, base_s, 0, base_size, pick_rand,
	                             num_iterations, allocator, NULL, NULL,
	                             NULL);
	assert(err == AML_SUCCESS);

	// Alloc increasing sizes and free random alloc.
	// We make sure not to exhaust all the memory pool, therefore,
	// this test should not fail.
	num_iterations = 1000;
	err = aml_alloc_workflow_run(max_s, base_s, 0, base_size, pick_rand,
	                             num_iterations, allocator, NULL, NULL,
	                             NULL);
	assert(err == AML_SUCCESS);

	// Alloc increasing sizes and free random alloc after 100 consecutive
	// allocs.
	// We make sure not to exhaust all the memory pool, therefore,
	// this test should not fail.
	num_iterations = 1000;
	err = aml_alloc_workflow_run(max_s, base_s, 100, base_size, pick_rand,
	                             num_iterations, allocator, NULL, NULL,
	                             NULL);
	assert(err == AML_SUCCESS);

	aml_allocator_sized_destroy(&allocator);
	aml_finalize();
	return 0;
}
