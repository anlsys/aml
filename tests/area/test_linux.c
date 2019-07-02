/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
*******************************************************************************/
#include "config.h"
#include "aml.h"
#include "aml/area/linux.h"
#include <assert.h>
#include <string.h>

const size_t sizes[3] = { 1, 1 << 12, 1 << 20 };

const int binding_flags[3] = {
	AML_AREA_LINUX_BINDING_FLAG_BIND,
	AML_AREA_LINUX_BINDING_FLAG_INTERLEAVE,
	AML_AREA_LINUX_BINDING_FLAG_PREFERRED
};

const int mmap_flags[2] = {
	AML_AREA_LINUX_MMAP_FLAG_PRIVATE,
	AML_AREA_LINUX_MMAP_FLAG_SHARED
};

void test_binding(struct aml_bitmap *bitmap)
{
	void *ptr;
	int binding_flag, mmap_flag;
	struct aml_area *area;

	for (unsigned int bf = 0;
	     bf < sizeof(binding_flags) / sizeof(*binding_flags); bf++) {
		binding_flag = binding_flags[bf];
		for (unsigned int mf = 0;
		     mf < sizeof(mmap_flags) / sizeof(*mmap_flags); mf++) {
			mmap_flag = mmap_flags[mf];
			for (size_t s = 0;
			     s < sizeof(sizes) / sizeof(*sizes); s++) {
				aml_area_linux_create(&area, mmap_flag, bitmap,
						      binding_flag);
				assert(area != NULL);
				ptr = area->ops->mmap(
				   (struct aml_area_data *)area->data,
				   NULL, sizes[s]);
				assert(ptr != NULL);
				memset(ptr, 0, sizes[s]);
				assert(aml_area_linux_check_binding(
				   (struct aml_area_linux_data *)area->data,
				   ptr, sizes[s]) > 0);
				assert(area->ops->munmap(
				   (struct aml_area_data *)area->data,
				   ptr, sizes[s]) == AML_SUCCESS);
				aml_area_linux_destroy(&area);
			}
		}

	}

}

void test_bind(void)
{
	struct bitmask *nodeset;
	int i, num_nodes;
	struct aml_bitmap bitmap;
	struct aml_area *area;

	nodeset = numa_get_mems_allowed();
	num_nodes = numa_bitmask_weight(nodeset);

	aml_bitmap_fill(&bitmap);
	if (aml_bitmap_last(&bitmap) > num_nodes) {
		assert(aml_area_linux_create
		       (&area, AML_AREA_LINUX_MMAP_FLAG_PRIVATE, &bitmap,
			AML_AREA_LINUX_BINDING_FLAG_PREFERRED) == -AML_EDOM);
		assert(area == NULL);
	}

	test_binding(NULL);

	aml_bitmap_zero(&bitmap);
	for (i = 0; i < num_nodes; i++) {
		aml_bitmap_set(&bitmap, i);
		test_binding(&bitmap);
		aml_bitmap_clear(&bitmap, i);
	}
}

int main(void)
{
	test_bind();
	return 0;
}
