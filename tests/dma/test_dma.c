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
