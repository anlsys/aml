/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
*******************************************************************************/
#include "aml.h"
#include "config.h"
#include "aml/area/linux.h"
#if RUN_CUDA
#include "aml/area/cuda.h"
#endif
#include <assert.h>

void test_map(const struct aml_area *area){
	assert(area != NULL);
	assert(area->ops->mmap != NULL);
	assert(area->ops->munmap != NULL);

	void *ptr;
	size_t s;
	const size_t sizes[4] = {1, 32, 4096, 1<<20};

	for(s = 0; s<sizeof(sizes)/sizeof(*sizes); s++){
		aml_errno = AML_SUCCESS;
		ptr = aml_area_mmap(area, sizes[s], NULL);
		assert(aml_errno == AML_SUCCESS);
		assert(ptr != NULL);
		assert(aml_area_munmap(area, ptr, sizes[s]) == AML_SUCCESS);
	}
}

void test_aml_area(struct aml_area *area){	
	test_map(area);
}

int main(int argc, char** argv){
	aml_init(&argc, &argv);
	test_map(&aml_area_linux);
#if RUN_CUDA	
	test_map(&aml_area_cuda);
#endif
	aml_finalize();
	return 0;
}
