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

#include "aml/area/linux.h"
#include "aml/dma/linux.h"
#include "aml/dma/multiplex.h"

#define PTR_OFF(ptr, sign, off) (void *)((intptr_t)(ptr)sign(intptr_t)(off))

int main(int argc, char *argv[])
{
	assert(aml_init(&argc, &argv) == AML_SUCCESS);

	struct aml_dma *linux1, *linux2;
	aml_dma_linux_create(&linux1, 1);
	aml_dma_linux_create(&linux2, 1);

	size_t size = 1UL << 20;
	const struct aml_dma *mydmas[] = {linux1, linux2};
	size_t myweights[] = {1, 1};
	aml_dma_operator myops[] = {aml_dma_linux_memcpy_op,
	                            aml_dma_linux_memcpy_op};
	size_t sizes[] = {size, size};
	struct aml_dma_multiplex_request_args myargs = {
	        .ops = myops,
	        .op_args = (void **)sizes,
	};

	struct aml_dma *dma;
	aml_dma_multiplex_create(&dma, 2, mydmas, myweights);

	void *src_buf = malloc(size);
	void *test_buf = malloc(size);
	void *dma_buf = aml_area_mmap(&aml_area_linux, size, NULL);

	assert(src_buf);
	assert(test_buf);
	assert(dma_buf);

	memset(src_buf, 1, size);
	memset(test_buf, 0, size);

	assert(aml_dma_copy_custom(dma, dma_buf, src_buf,
	                           aml_dma_multiplex_copy_single,
	                           &myargs) == AML_SUCCESS);
	assert(aml_dma_copy_custom(dma, test_buf, dma_buf,
	                           aml_dma_multiplex_copy_single,
	                           &myargs) == AML_SUCCESS);

	assert(!memcmp(test_buf, src_buf, size));

	free(src_buf);
	free(test_buf);
	aml_area_munmap(&aml_area_linux, dma_buf, size);

	const size_t n = 128;
	size = 1UL << 10;
	void *dma_bufs[n];

	src_buf = malloc(size * n);
	test_buf = malloc(size * n);
	assert(src_buf);
	assert(test_buf);
	memset(test_buf, 0, size * n);

	sizes[0] = size;
	sizes[1] = size;

	for (size_t i = 0; i < n; i++) {
		memset(PTR_OFF(src_buf, +, i * size), i + 1, size);
		dma_bufs[i] = aml_area_mmap(&aml_area_linux, size, NULL);
		assert(dma_bufs[i]);
	}

// Copy to device area
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (size_t i = 0; i < n; i++) {
		assert(aml_dma_async_copy_custom(dma, NULL, dma_bufs[i],
		                                 PTR_OFF(src_buf, +, size * i),
		                                 aml_dma_multiplex_copy_single,
		                                 &myargs) == AML_SUCCESS);
	}
	// Wait all copies
	assert(aml_dma_barrier(dma) == AML_SUCCESS);

// Copy back from device area to host
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (size_t i = 0; i < n; i++) {
		assert(aml_dma_async_copy_custom(
		               dma, NULL, PTR_OFF(test_buf, +, size * i),
		               dma_bufs[i], aml_dma_multiplex_copy_single,
		               &myargs) == AML_SUCCESS);
	}
	// Wait all copies
	assert(aml_dma_barrier(dma) == AML_SUCCESS);

	// Byte wise comparison
	for (size_t i = 0; i < n; i++)
		assert(!memcmp(PTR_OFF(test_buf, +, size * i),
		               PTR_OFF(src_buf, +, size * i), size));

	// Cleanup
	free(src_buf);
	free(test_buf);
	for (size_t i = 0; i < n; i++)
		aml_area_munmap(&aml_area_linux, dma_bufs[i], size);

	aml_finalize();
}
