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
#include "aml/higher/allocator/utils.h"

void test_chunk_match()
{
	struct aml_memory_chunk a, b;

	// Match ok
	a.ptr = (void *)0;
	a.size = 8;
	b.ptr = (void *)8;
	b.size = 0;
	assert(aml_memory_chunk_match(a, b));
	assert(aml_memory_chunk_match(b, a));

	// Match not ok
	b.ptr = (void *)16;
	b.size = 0;
	assert(!aml_memory_chunk_match(a, b));
	assert(!aml_memory_chunk_match(b, a));

	b.ptr = (void *)4;
	b.size = 0;
	assert(!aml_memory_chunk_match(a, b));
	assert(!aml_memory_chunk_match(b, a));
}

void test_chunk_overlap()
{
	struct aml_memory_chunk a, b;

	// Overlapping
	a.ptr = (void *)0;
	a.size = 8;
	b.ptr = (void *)0;
	b.size = 8;
	assert(aml_memory_chunk_overlap(a, b));
	assert(aml_memory_chunk_overlap(b, a));

	b.ptr = (void *)4;
	b.size = 8;
	assert(aml_memory_chunk_overlap(a, b));
	assert(aml_memory_chunk_overlap(b, a));

	// Non overlapping
	b.ptr = (void *)8;
	b.size = 0;
	assert(!aml_memory_chunk_overlap(a, b));
	assert(!aml_memory_chunk_overlap(b, a));

	b.ptr = (void *)16;
	b.size = 0;
	assert(!aml_memory_chunk_overlap(a, b));
	assert(!aml_memory_chunk_overlap(b, a));
}

void test_chunk_contains()
{
	struct aml_memory_chunk a, b;

	// Contains
	a.ptr = (void *)16;
	a.size = 8;
	b.ptr = (void *)16;
	b.size = 8;
	assert(aml_memory_chunk_contains(a, b));

	b.ptr = (void *)16;
	b.size = 4;
	assert(aml_memory_chunk_contains(a, b));

	b.ptr = (void *)23;
	b.size = 1;
	assert(aml_memory_chunk_contains(a, b));

	b.ptr = (void *)20;
	b.size = 2;
	assert(aml_memory_chunk_contains(a, b));

	// Does not contain
	b.ptr = (void *)24;
	b.size = 1;
	assert(!aml_memory_chunk_contains(a, b));

	b.ptr = (void *)23;
	b.size = 2;
	assert(!aml_memory_chunk_contains(a, b));

	b.ptr = (void *)15;
	b.size = 1;
	assert(!aml_memory_chunk_contains(a, b));

	b.ptr = (void *)15;
	b.size = 2;
	assert(!aml_memory_chunk_contains(a, b));

	b.ptr = (void *)15;
	b.size = 10;
	assert(!aml_memory_chunk_contains(a, b));
}

void test_memory_pool()
{
	const size_t size = 1UL << 10;
	struct aml_memory_pool *pool;
	void *ptr = (void *)0x7fffabcd0000;

	// Init
	assert(aml_memory_pool_create(&pool, ptr, size) == AML_SUCCESS);

	// Push out of bound.
	assert(aml_memory_pool_push(
	               pool,
	               (void *)((intptr_t)pool->memory.ptr + pool->memory.size),
	               1) == -AML_EDOM);
	assert(aml_memory_pool_push(pool, pool->memory.ptr,
	                            pool->memory.size * 2) == -AML_EDOM);

	// Push overlap existing chunk
	for (size_t i = 0; i < size; i += size / 8)
		assert(aml_memory_pool_push(
		               pool, (void *)((intptr_t)pool->memory.ptr + i),
		               size / 8) == -AML_EINVAL);

	// Pop more than available
	assert(aml_memory_pool_pop(pool, &ptr, pool->memory.size * 2) ==
	       -AML_ENOMEM);

	// Pop everything
	assert(aml_memory_pool_pop(pool, &ptr, pool->memory.size) ==
	       AML_SUCCESS);

	// Pop anything fails because pool is empty.
	assert(aml_memory_pool_pop(pool, &ptr, 1) == -AML_ENOMEM);

	// Fragment the pool by reinserting chunks of 1<<4 size spaced by one
	// chunk.
	for (size_t i = 0; i < size; i += 1 << 5)
		assert(aml_memory_pool_push(
		               pool, (void *)((intptr_t)pool->memory.ptr + i),
		               1 << 4) == AML_SUCCESS);

	// Popping a chunk bigger than 1<<4 does not work.
	assert(aml_memory_pool_pop(pool, &ptr, 1 + (1 << 4)) == -AML_ENOMEM);

	// Insert complementary chunks.
	for (size_t i = 1 << 4; i < size; i += 1 << 5)
		assert(aml_memory_pool_push(
		               pool, (void *)((intptr_t)pool->memory.ptr + i),
		               1 << 4) == AML_SUCCESS);

	// Pop everything should work.
	assert(aml_memory_pool_pop(pool, &ptr, pool->memory.size) ==
	       AML_SUCCESS);

	// Cleanup
	assert(aml_memory_pool_destroy(&pool) == AML_SUCCESS);
}

int main(int argc, char **argv)
{
	assert(aml_init(&argc, &argv) == AML_SUCCESS);

	test_chunk_match();
	test_chunk_overlap();
	test_chunk_contains();
	test_memory_pool();

	aml_finalize();
}
