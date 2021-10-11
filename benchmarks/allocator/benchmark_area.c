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
#include "../../tests/allocator/memset.h"
#include "../tests/allocator/dummy_area.h"

#include "aml/higher/allocator.h"
#include "aml/higher/allocator/area.h"

#include "allocator.h"

void benchmark_area_allocator(FILE *out,
                              struct aml_area *area,
                              const char *area_name,
                              aml_memset_fn memset_fn)
{
	const size_t num_samples = 20;
	struct aml_allocator *allocator;
	char allocator_name[strlen(area_name) + strlen("area_") + 1];

	// Set allocator name
	snprintf(allocator_name, sizeof(allocator_name), "area_%s", area_name);
	allocator_name[sizeof(allocator_name) - 1] = '\0';

	// Run benchmarks
	// 1.
	assert(aml_allocator_area_create(&allocator, area, NULL) ==
	       AML_SUCCESS);
	for (size_t i = 0; i < num_samples; i++)
		benchmark_consecutive_allocations(out, allocator_name,
		                                  allocator, memset_fn);
	aml_allocator_area_destroy(&allocator);

	// 2.
	assert(aml_allocator_area_create(&allocator, area, NULL) ==
	       AML_SUCCESS);
	for (size_t i = 0; i < num_samples; i++)
		benchmark_consecutive_allocations_free(out, allocator_name,
		                                       allocator, memset_fn);
	aml_allocator_area_destroy(&allocator);

	// 3.
	assert(aml_allocator_area_create(&allocator, area, NULL) ==
	       AML_SUCCESS);
	for (size_t i = 0; i < num_samples; i++)
		benchmark_random_allocations_free(out, allocator_name,
		                                  allocator, memset_fn);
	aml_allocator_area_destroy(&allocator);
}

int main(int argc, char **argv)
{
	assert(aml_init(&argc, &argv) == AML_SUCCESS);

	// Benchmark dummy area to measure library overhead.
	benchmark_area_allocator(stderr, &aml_area_dummy, "dummy",
	                         aml_dummy_memset);

	// Benchmark linux area
	benchmark_area_allocator(stderr, &aml_area_linux, "linux",
	                         aml_linux_memset);

	// Benchmark cuda area
#if AML_HAVE_BACKEND_CUDA == 1
	if (aml_support_backends(AML_BACKEND_CUDA))
		benchmark_area_allocator(stderr, &aml_area_cuda, "cuda",
		                         aml_cuda_memset);
#endif

		// Benchmark ze area
#if AML_HAVE_BACKEND_ZE == 1
	if (aml_support_backends(AML_BACKEND_ZE))
		benchmark_area_allocator(stderr, aml_area_ze_device, "ze",
		                         aml_ze_memset);
#endif

	aml_finalize();
	return 0;
}
