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

#include "aml/area/linux.h"
#if AML_HAVE_BACKEND_CUDA == 1
#include "aml/area/cuda.h"
#endif
#if AML_HAVE_BACKEND_ZE == 1
#include "aml/area/ze.h"
#endif
#if AML_HAVE_BACKEND_HIP == 1
#include "aml/area/hip.h"
#endif
#include "../tests/allocator/dummy_area.h"

#include "aml/higher/allocator.h"
#include "aml/higher/allocator/buddy.h"

#include "allocator.h"

void benchmark_buddy_allocator(FILE *out,
                               struct aml_area *area,
                               const char *area_name)
{
	const size_t num_samples = 20;
	struct aml_allocator *allocator;
	char allocator_name[strlen(area_name) + strlen("buddy_") + 1];

	// Set allocator name
	snprintf(allocator_name, sizeof(allocator_name), "buddy_%s", area_name);
	allocator_name[sizeof(allocator_name) - 1] = '\0';

	// Run benchmarks
	// 1.
	assert(aml_allocator_buddy_create(&allocator, area, NULL) ==
	       AML_SUCCESS);
	for (size_t i = 0; i < num_samples; i++)
		benchmark_consecutive_allocations(out, allocator_name,
		                                  allocator);
	aml_allocator_buddy_destroy(&allocator);

	// 2.
	assert(aml_allocator_buddy_create(&allocator, area, NULL) ==
	       AML_SUCCESS);
	for (size_t i = 0; i < num_samples; i++)
		benchmark_consecutive_allocations_free(out, allocator_name,
		                                       allocator);
	aml_allocator_buddy_destroy(&allocator);

	// 3.
	assert(aml_allocator_buddy_create(&allocator, area, NULL) ==
	       AML_SUCCESS);
	for (size_t i = 0; i < num_samples; i++)
		benchmark_random_allocations_free(out, allocator_name,
		                                  allocator);
	aml_allocator_buddy_destroy(&allocator);
}

int main(int argc, char **argv)
{
	assert(aml_init(&argc, &argv) == AML_SUCCESS);

	// Benchmark dummy area to measure library overhead.
	benchmark_buddy_allocator(stderr, &aml_area_dummy, "dummy");

	// Benchmark linux area
	benchmark_buddy_allocator(stderr, &aml_area_linux, "linux");

	// Benchmark cuda area
#if AML_HAVE_BACKEND_CUDA == 1
	if (aml_support_backends(AML_BACKEND_CUDA))
		benchmark_buddy_allocator(stderr, &aml_area_cuda, "cuda");
#endif

		// Benchmark ze area
#if AML_HAVE_BACKEND_ZE == 1
	if (aml_support_backends(AML_BACKEND_ZE))
		benchmark_buddy_allocator(stderr, aml_area_ze_device, "ze");
#endif

		// Benchmark hip area
#if AML_HAVE_BACKEND_HIP == 1
	if (aml_support_backends(AML_BACKEND_HIP))
		benchmark_buddy_allocator(stderr, &aml_area_hip, "hip");
#endif

	aml_finalize();
	return 0;
}
