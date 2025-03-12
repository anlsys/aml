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

#include <assert.h>

#include "aml.h"

#include "aml/higher/allocator.h"

#include "../benchmarks/utils.h"

static size_t greater_power_of_two(size_t size)
{
	size_t i = 0;
	while ((1UL << i) < size)
		++i;
	return (1UL << i);
}

int aml_alloc_workflow_chunk_run(struct aml_allocator *allocator)
{
#define N 1024
	struct aml_allocator_chunk *chunks[N];
	memset(chunks, 0, sizeof(chunks));

	for (int i = 0; i < N; ++i) {
#define MIN_SIZE (1)
#define MAX_SIZE (64 * 1024 * 1024)
		const size_t size =
		        MIN_SIZE + (rand() % (MAX_SIZE - MIN_SIZE + 1));
		chunks[i] = aml_allocator_alloc_chunk(allocator, size);
		assert(chunks[i]);
		assert(chunks[i]->ptr);
		assert(chunks[i]->size >= size);
		assert(chunks[i]->size <= greater_power_of_two(size));

		if (rand() % 4 == 0) {
			int idx;
			do {
				idx = rand() % i;
			} while (chunks[idx] == NULL);
			aml_allocator_free_chunk(allocator, chunks[idx]);
			chunks[idx] = NULL;
		}
	}

	return AML_SUCCESS;
}
