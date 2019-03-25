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
#include "aml/area/area.h"
#include "config.h"
#ifdef HAVE_HWLOC
#include "aml/area/hwloc.h"
#endif
	
void test_binding(struct aml_area *area){
	void *ptr;
	int err;
	size_t s;
	const size_t sizes[2] = {1, 1<<20};

	for(s = 0; s<sizeof(sizes)/sizeof(*sizes); s++){
		ptr = NULL;
		err = area->ops->map(area, &ptr, sizes[s]);
		if(err == AML_AREA_ENOMEM)
			continue;
		if(err == AML_AREA_EINVAL)
			continue;
		if(err == AML_SUCCESS){
			memset(ptr, 0, sizes[s]);
			if(area->ops->check_binding)
				assert(area->ops->check_binding(area, ptr, sizes[s]) > 0);
			assert(area->ops->unmap(area, ptr, sizes[s]) == AML_SUCCESS);
		} else
			assert(0);
	}

}

void test_bind(struct aml_area *area){
	if(area->ops->bind == NULL)
		return;

	int err;
	size_t i,j;
	unsigned long flags = 1;
	struct aml_bitmap bitmap;
	
	err = area->ops->bind(area, NULL, 0);
	if(err == AML_SUCCESS)
		test_binding(area);

	for(i = 0; i < sizeof(flags)*8 + 1; i++){
		for(j = 0; j<AML_BITMAP_MAX; j++){
			aml_bitmap_zero(&bitmap);
			aml_bitmap_set(&bitmap, j);
			err = area->ops->bind(area, &bitmap, flags);			
			if(err == AML_SUCCESS)
				test_binding(area);
		}
		flags = flags << 1;
	}
}

void test_malloc(struct aml_area *area){
        if(area->ops->malloc == NULL)
		return;
	assert(area->ops->free != NULL);

	void *ptr;
	int err;
	size_t s, a;
	const size_t sizes[4]       = {1, 32, 4096, 1<<20};
	const size_t alignements[4] = {1, 32, 13,   1<<20};
	
	for(s = 0; s<sizeof(sizes)/sizeof(*sizes); s++){
		for(a = 0; a<sizeof(alignements)/sizeof(*alignements); a++){
			ptr = NULL;
			err = area->ops->malloc(area, &ptr, sizes[s], alignements[a]);
			if(err == AML_AREA_ENOTSUP)
			        return;
			if(err == AML_AREA_ENOMEM)
				continue;
			if(err == AML_AREA_EINVAL)
				continue;
			if(err == AML_SUCCESS){
				memset(ptr, 0, sizes[s]);
				assert(area->ops->free(area, ptr) == AML_SUCCESS);
			} else
				assert(0);
		}
	}
}

void test_map(struct aml_area *area){
	assert(area->ops->map != NULL);
	assert(area->ops->unmap != NULL);

	void *ptr;
	int err;
	size_t s;
	const size_t sizes[4] = {1, 32, 4096, 1<<20};

	for(s = 0; s<sizeof(sizes)/sizeof(*sizes); s++){
		ptr = NULL;
		err = area->ops->map(area, &ptr, sizes[s]);
		if(err == AML_AREA_ENOMEM)
			continue;
		if(err == AML_AREA_EINVAL)
			continue;
		if(err == AML_SUCCESS){
			memset(ptr, 0, sizes[s]);
			assert(area->ops->unmap(area, ptr, sizes[s]) == AML_SUCCESS);
		} else
			assert(0);
	}
}

void test_create(struct aml_area *area){
	if(area->ops->create == NULL)
		return;

	assert(area->ops->destroy != NULL);
	
	struct aml_area new;
	int err;

	new.ops = area->ops;
	new.data = NULL;
	assert(area->ops->create(&new) == AML_SUCCESS);
	area->ops->destroy(&new);
}

void test_aml_area(struct aml_area *area){	
	test_create(area);
	test_map(area);
	test_malloc(area);
	test_bind(area);
}

int main(int argc, char** argv){
	aml_init(&argc, &argv);
	test_aml_area(aml_area_linux_private);
	test_aml_area(aml_area_linux_shared);
#ifdef HAVE_HWLOC
	test_aml_area(aml_area_hwloc_private);
	test_aml_area(aml_area_hwloc_shared);
#endif
	aml_finalize();
	return 0;
}

