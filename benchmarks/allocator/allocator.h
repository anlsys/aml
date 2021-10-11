/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include "../../tests/allocator/memset.h"

#include <stdio.h>

#include "aml/higher/allocator.h"

// Alloc consecutively pages of 4KiB until 4GiB of memory is allocated.
void benchmark_consecutive_allocations(FILE *out,
                                       const char *allocator_name,
                                       struct aml_allocator *allocator,
                                       aml_memset_fn memset_fn);

// Alloc consecutively pages of 4KiB and free them after each allocation.
void benchmark_consecutive_allocations_free(FILE *out,
                                            const char *allocator_name,
                                            struct aml_allocator *allocator,
                                            aml_memset_fn memset_fn);

// Alloc consecutively 2^9 chunks of random sizes from 4KiB to 16MiB.
// Then for each new allocation of a random size in that range, one
// previous random allocation is freed.
void benchmark_random_allocations_free(FILE *out,
                                       const char *allocator_name,
                                       struct aml_allocator *allocator,
                                       aml_memset_fn memset_fn);
