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
#include "aml/area/linux.h"

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

