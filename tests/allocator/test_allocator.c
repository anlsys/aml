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

#include "aml.h"

#include "aml/higher/allocator.h"
#include "internal/utarray.h"

#include "../benchmarks/utils.h"

int aml_alloc_record_next_size(struct aml_alloc_record *record, size_t *size)
{
	if (record == NULL || size == NULL)
		return -AML_EINVAL;

	int err = record->next_size(record, size);
	if (err != AML_SUCCESS)
		return err;

	record->counter++;
	return AML_SUCCESS;
}

int rand_size(const struct aml_alloc_record *record, size_t *size)
{
	size_t s = rand() % record->max_size;
	s = s - (s % record->base_size);
	if (s > 0)
		*size = s;
	else
		*size = record->base_size;
	return AML_SUCCESS;
}

int base_size(const struct aml_alloc_record *record, size_t *size)
{
	*size = record->base_size;
	return AML_SUCCESS;
}

int max_size(const struct aml_alloc_record *record, size_t *size)
{
	*size = record->max_size;
	return AML_SUCCESS;
}

int increasing_size(const struct aml_alloc_record *record, size_t *size)
{

	size_t s = (record->base_size * record->counter) % record->max_size;
	*size = record->base_size + s;
	return AML_SUCCESS;
}

int aml_alloc_record_next_free(const struct aml_alloc_record *record,
                               size_t *index)
{

	if (record == NULL || index == NULL)
		return -AML_EINVAL;

	if (record->counter < record->delay)
		return -AML_EDOM;

	UT_array *ptrs = (UT_array *)record->ptrs;
	const size_t len = utarray_len(ptrs);
	if (len == 0)
		return -AML_EDOM;

	size_t i;
	int err = record->next_free(record, &i);

	if (err != AML_SUCCESS)
		return err;

	*index = i % len;

	return AML_SUCCESS;
}

int no_pick(const struct aml_alloc_record *record, size_t *out)
{
	(void)out;
	(void)record;
	return -AML_EDOM;
}

int pick_prev(const struct aml_alloc_record *record, size_t *out)
{
	if (record->counter < record->delay)
		return -AML_EDOM;
	*out = (record->counter - record->delay);
	return AML_SUCCESS;
}

int pick_rand(const struct aml_alloc_record *record, size_t *out)
{
	(void)record;
	if (record->counter < record->delay)
		return -AML_EDOM;
	*out = rand();
	return AML_SUCCESS;
}

struct aml_memory_chunk {
	void *ptr;
	size_t size;
};

UT_icd ptr_icd = {
        .sz = sizeof(struct aml_memory_chunk),
        .init = NULL,
        .copy = NULL,
        .dtor = NULL,
};

/**
 * Allocate an empty record. The record can be released later with `free()`.
 * @param[in] max_size: Maximum allocation size.
 * @param[in] base_size: Allocation base size. Allocation sizes will be a
 * multiple of this number.
 * @param[in] delay: Number of allocations before freeing any allocations.
 * @param[in] next_size: Function returning the size of the next allocation.
 * @param[in] next_free: Function returning the slot of the next free.
 * @return A newly allocated record. This function asserts that the record
 * is allocated.
 */
static struct aml_alloc_record *
aml_alloc_record_create(const size_t max_size,
                        const size_t base_size,
                        const size_t delay,
                        aml_alloc_record_next_size_fn next_size,
                        aml_alloc_record_next_free_fn next_free)
{
	struct aml_alloc_record *test = malloc(sizeof(*test));
	assert(test != NULL);

	UT_array *ptrs;
	utarray_new(ptrs, &ptr_icd);
	test->ptrs = ptrs;

	test->max_size = max_size - (max_size % base_size);
	test->base_size = base_size;
	test->delay = delay;
	test->counter = 0;
	test->next_size = next_size;
	test->next_free = next_free;
	return test;
}

static void aml_alloc_record_destroy(struct aml_alloc_record *record,
                                     struct aml_allocator *allocator)
{

	// Empty queue
	struct aml_memory_chunk *c;
	UT_array *ptrs = (UT_array *)record->ptrs;

	for (size_t i = 0; i < utarray_len(ptrs); i++) {
		c = (struct aml_memory_chunk *)utarray_eltptr(ptrs, i);
		assert(aml_free(allocator, c->ptr) == AML_SUCCESS);
	}

	// Cleanup record.
	utarray_free(ptrs);
	free(record);
}

static void *record_alloc(struct aml_allocator *allocator,
                          const size_t size,
                          struct aml_time_stats *stats)
{
	aml_time_t ts, te;
	aml_gettime(&ts);
	void *ptr = aml_alloc(allocator, size);
	aml_gettime(&te);

	if (ptr != NULL && stats != NULL)
		aml_time_stats_add(stats, aml_timediff(ts, te));
	return ptr;
}

static void record_free(struct aml_allocator *allocator,
                        void *ptr,
                        struct aml_time_stats *stats)
{
	aml_time_t ts, te;

	aml_gettime(&ts);
	assert(aml_free(allocator, ptr) == AML_SUCCESS);
	aml_gettime(&te);

	if (stats != NULL)
		aml_time_stats_add(stats, aml_timediff(ts, te));
}

void aml_alloc_workflow_output_print(FILE *out,
                                     struct aml_alloc_workflow_output o)
{
	fprintf(out, "Total size:                      %lu\n", o.total_size);
	fprintf(out, "Still allocated size:            %lu\n",
	        o.allocated_size);
	fprintf(out, "Maximum allocated size:          %lu\n",
	        o.max_allocated_size);
	fprintf(out, "Number of allocations:           %lu\n", o.num_alloc);
	fprintf(out, "Number of frees:                 %lu\n", o.num_free);
	fprintf(out, "Last allocation size:            %lu\n", o.last_size);
	fprintf(out, "Number of successful iterations: %lu\n",
	        o.num_iterations);
}

int aml_alloc_workflow_run(const size_t max_size,
                           const size_t base_size,
                           const size_t delay,
                           aml_alloc_record_next_size_fn next_size,
                           aml_alloc_record_next_free_fn next_free,
                           const size_t num_iterations,
                           struct aml_allocator *allocator,
                           struct aml_time_stats *alloc_stats,
                           struct aml_time_stats *free_stats,
                           struct aml_alloc_workflow_output *out)
{
	int err;
	struct aml_alloc_workflow_output output;
	size_t size, spot, i;
	size_t total_size = 0;
	size_t allocated_size = 0;
	size_t max_allocated_size = 0;
	size_t num_alloc = 0;
	size_t num_free = 0;

	if (base_size == 0 || next_size == NULL || next_free == NULL ||
	    num_iterations == 0 || allocator == NULL)
		return -AML_EINVAL;

	struct aml_alloc_record *rec = aml_alloc_record_create(
	        max_size, base_size, delay, next_size, next_free);
	UT_array *ptrs = (UT_array *)rec->ptrs;

	for (i = 0; i < num_iterations; i++) {
		err = aml_alloc_record_next_size(rec, &size);
		assert(err == AML_SUCCESS);

		err = aml_alloc_record_next_free(rec, &spot);
		assert(err == AML_SUCCESS || err == -AML_EDOM);

		// Do allocation.
		void *ptr = record_alloc(allocator, size, alloc_stats);
		if (ptr == NULL) {
			assert(aml_errno != AML_SUCCESS);
			err = -aml_errno;
			goto cleanup;
		}

		struct aml_memory_chunk c = {.ptr = ptr, .size = size};
		utarray_push_back(ptrs, (void *)&c);
		total_size += size;
		allocated_size += size;
		max_allocated_size = allocated_size > max_allocated_size ?
		                             allocated_size :
		                             max_allocated_size;
		num_alloc += 1;

		// Do free. If output of aml_alloc_record_next_free was
		// -AML_EDOM, Then we don't free.
		if (err == -AML_EDOM)
			continue;

		struct aml_memory_chunk *c_ptr =
		        (struct aml_memory_chunk *)utarray_eltptr(ptrs, spot);
		record_free(allocator, c_ptr->ptr, free_stats);
		allocated_size -= c_ptr->size;
		num_free += 1;
		utarray_erase(ptrs, spot, 1);
	}
	err = AML_SUCCESS;

	// Cleanup
cleanup:
	output.total_size = total_size;
	output.allocated_size = allocated_size;
	output.max_allocated_size = max_allocated_size;
	output.num_alloc = num_alloc;
	output.num_free = num_free;
	output.last_size = size;
	output.num_iterations = i;

	if (err != AML_SUCCESS)
		aml_alloc_workflow_output_print(stderr, output);
	if (out != NULL)
		memcpy(out, &output, sizeof(*out));
	aml_alloc_record_destroy(rec, allocator);
	return err;
}
