/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *******************************************************************************/

#include "aml.h"

#define PTR_OFF(ptr, sign, off) (void *)((intptr_t)(ptr)sign(intptr_t)(off))

void test_dma_memcpy(struct aml_area *area,
                     struct aml_area_mmap_options *area_opts,
                     struct aml_dma *dma,
                     aml_dma_operator memcpy_op)
{
	const size_t size = 1UL << 20; // 1MiB
	void *src_buf, *test_buf, *dma_buf;

	src_buf = malloc(size);
	test_buf = malloc(size);
	dma_buf = aml_area_mmap(area, size, area_opts);

	assert(src_buf);
	assert(test_buf);
	assert(dma_buf);

	memset(src_buf, 1, size);
	memset(test_buf, 0, size);

	assert(aml_dma_copy_custom(dma, dma_buf, src_buf, memcpy_op,
	                           (void *)size) == AML_SUCCESS);
	assert(aml_dma_copy_custom(dma, test_buf, dma_buf, memcpy_op,
	                           (void *)size) == AML_SUCCESS);
	assert(!memcmp(test_buf, src_buf, size));

	free(src_buf);
	free(test_buf);
	aml_area_munmap(area, dma_buf, size);
}

void test_dma_sync(struct aml_area *area,
                   struct aml_area_mmap_options *area_opts,
                   struct aml_dma *dma,
                   aml_dma_operator memcpy_op)
{
	const size_t n = 128;
	const size_t size = 1UL << 10; // 1KiB
	void *src_buf, *test_buf, *dma_buf[n];

	// Initialization
	src_buf = malloc(size * n);
	test_buf = malloc(size * n);
	assert(src_buf);
	assert(test_buf);
	memset(test_buf, 0, size * n);
	for (size_t i = 0; i < n; i++) {
		memset(PTR_OFF(src_buf, +, i * size), i + 1, size);
		dma_buf[i] = aml_area_mmap(area, size, area_opts);
		assert(dma_buf[i]);
	}

	// Copy to device area
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (size_t i = 0; i < n; i++) {
		assert(aml_dma_async_copy_custom(dma, NULL, dma_buf[i],
		                                 PTR_OFF(src_buf, +, size * i),
		                                 memcpy_op,
		                                 (void *)size) == AML_SUCCESS);
	}
	// Wait all copies
	assert(aml_dma_sync(dma) == AML_SUCCESS);

	// Copy back from device area to host
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (size_t i = 0; i < n; i++) {
		assert(aml_dma_async_copy_custom(dma, NULL,
		                                 PTR_OFF(test_buf, +, size * i),
		                                 dma_buf[i], memcpy_op,
		                                 (void *)size) == AML_SUCCESS);
	}
	// Wait all copies
	assert(aml_dma_sync(dma) == AML_SUCCESS);

	// Byte wise comparison
	for (size_t i = 0; i < n; i++)
		assert(!memcmp(PTR_OFF(test_buf, +, size * i),
		               PTR_OFF(src_buf, +, size * i), size));

	// Cleanup
	free(src_buf);
	free(test_buf);
	for (size_t i = 0; i < n; i++)
		aml_area_munmap(area, dma_buf[i], size);
}
