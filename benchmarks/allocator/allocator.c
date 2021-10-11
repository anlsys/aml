/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include "../tests/allocator/test_allocator.h"

#include "aml.h"

#include "aml/area/linux.h"
#if AML_HAVE_BACKEND_CUDA == 1
#include "aml/area/cuda.h"
#endif
#if AML_HAVE_BACKEND_ZE == 1
#include "aml/area/ze.h"
#endif
#include "../tests/allocator/dummy_area.h"

#include "aml/higher/allocator.h"

// Struct with benchmark information to print to file.
struct benchmark_sample {
	const char *benchmark_name;
	const char *allocator_name;
	struct aml_time_stats alloc_stats;
	struct aml_time_stats free_stats;
};

// Print benchmark header.
void fprint_benchmark_header(FILE *out)
{
	fprintf(out, "%s %s %s %s %s %s %s %s\n", "Benchmark", "Allocator",
	        "alloc_min", "alloc_mean", "alloc_max", "free_min", "free_mean",
	        "free_max");
}

// Print a benchmark result.
void fprint_benchmark_sample(FILE *out, struct benchmark_sample *sample)
{
	fprintf(out, "%s %s %lld %f %lld %lld %f %lld\n",
	        sample->benchmark_name, sample->allocator_name,
	        sample->alloc_stats.min,
	        (double)sample->alloc_stats.sum / (double)sample->alloc_stats.n,
	        sample->alloc_stats.max, sample->free_stats.min,
	        (double)sample->free_stats.sum / (double)sample->free_stats.n,
	        sample->free_stats.max);
}

//------------------------------------------------------------------------//
// Benchmarks declaration.
//------------------------------------------------------------------------//

// Alloc consecutively pages of 4KiB until 4GiB of memory is allocated.
void benchmark_consecutive_allocations(FILE *out,
                                       const char *allocator_name,
                                       struct aml_allocator *allocator,
                                       aml_memset_fn memset_fn)
{
	const size_t base_s = 1UL << 12; // 4KiB
	const size_t max_s = base_s;
	size_t num_iterations = 1UL << 10; // 4MiB Total
	const size_t delay = 0;
	aml_alloc_record_next_size_fn size_fn = base_size;
	aml_alloc_record_next_free_fn next_fn = no_pick;

	struct aml_time_stats alloc_stats, free_stats;
	aml_stats_init(&alloc_stats);
	aml_stats_init(&free_stats);

	assert(aml_alloc_workflow_run(max_s, base_s, delay, size_fn, next_fn,
	                              num_iterations, allocator, memset_fn,
	                              &alloc_stats, &free_stats,
	                              NULL) == AML_SUCCESS);

	struct benchmark_sample sample = {
	        .alloc_stats = alloc_stats,
	        .free_stats = free_stats,
	        .benchmark_name = "consecutive_allocations",
	        .allocator_name = allocator_name,
	};

	fprint_benchmark_sample(out, &sample);
}

// Alloc consecutively pages of 4KiB and free them after each allocation.
void benchmark_consecutive_allocations_free(FILE *out,
                                            const char *allocator_name,
                                            struct aml_allocator *allocator,
                                            aml_memset_fn memset_fn)
{
	const size_t base_s = 1UL << 12; // 4KiB
	const size_t max_s = base_s;
	size_t num_iterations = 1UL << 10;
	const size_t delay = 0;
	aml_alloc_record_next_size_fn size_fn = base_size;
	aml_alloc_record_next_free_fn next_fn = pick_prev;

	struct aml_time_stats alloc_stats, free_stats;
	aml_stats_init(&alloc_stats);
	aml_stats_init(&free_stats);

	assert(aml_alloc_workflow_run(max_s, base_s, delay, size_fn, next_fn,
	                              num_iterations, allocator, memset_fn,
	                              &alloc_stats, &free_stats,
	                              NULL) == AML_SUCCESS);

	struct benchmark_sample sample = {
	        .alloc_stats = alloc_stats,
	        .free_stats = free_stats,
	        .benchmark_name = "4KiB_allocations_free",
	        .allocator_name = allocator_name,
	};
	fprint_benchmark_sample(out, &sample);
}

// Alloc consecutively 2^9 chunks of random sizes from 4KiB to 16MiB.
// Then for each new allocation of a random size in that range, one
// previous random allocation is freed.
void benchmark_random_allocations_free(FILE *out,
                                       const char *allocator_name,
                                       struct aml_allocator *allocator,
                                       aml_memset_fn memset_fn)
{
	// Make this function deterministic and identical on each call.
	srand(0);
	const size_t base_s = 1UL << 12; // 4KiB
	const size_t max_s = 1UL << 24; // 16MiB
	size_t num_iterations = 1UL << 10;
	const size_t delay = 1UL << 9; // [2MiB, 8GiB]
	aml_alloc_record_next_size_fn size_fn = rand_size;
	aml_alloc_record_next_free_fn next_fn = pick_rand;

	struct aml_time_stats alloc_stats, free_stats;
	aml_stats_init(&alloc_stats);
	aml_stats_init(&free_stats);

	assert(aml_alloc_workflow_run(max_s, base_s, delay, size_fn, next_fn,
	                              num_iterations, allocator, memset_fn,
	                              &alloc_stats, &free_stats,
	                              NULL) == AML_SUCCESS);

	struct benchmark_sample sample = {
	        .alloc_stats = alloc_stats,
	        .free_stats = free_stats,
	        .benchmark_name = "4KiB-16MiB_rand_allocations_free",
	        .allocator_name = allocator_name,
	};
	fprint_benchmark_sample(out, &sample);
}
