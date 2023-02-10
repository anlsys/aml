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

int main(int argc, char *argv[])
{
	assert(aml_init(&argc, &argv) == AML_SUCCESS);

	struct aml_dma *linux1, *linux2;
	aml_dma_linux_create(&linux1, 1);
	aml_dma_linux_create(&linux2, 1);

	const size_t size = 1UL << 20;
	const struct aml_dma *mydmas[] = { linux1, linux2 };
	size_t myweights[] = { 1, 1 };
	aml_dma_operator myops[] = { aml_dma_linux_memcpy_op,
		aml_dma_linux_memcpy_op };
	size_t sizes[] = { size, size };
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

	aml_finalize();
}
