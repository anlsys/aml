/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
*******************************************************************************/

#include <assert.h>
#include "aml.h"
#include <sys/mman.h>
#include <numaif.h>

/*******************************************************************************
 * mbind methods for Linux systems
 * Only handles the actual mbind/mempolicy calls
 ******************************************************************************/

/* common to both methods */
int aml_area_linux_mbind_generic_binding(const struct aml_area_linux_mbind_data *data,
					 struct aml_binding **b)
{
	assert(data != NULL);
	const struct aml_bitmap *nodemask = &data->nodemask;
	/* not exactly proper, we should inspect the nodemask to find the real
	 * binding policy.
	 */
	if(data->policy == MPOL_BIND)
	{
		for(int i = 0; i < AML_BITMAP_MAX; i++)
			if(aml_bitmap_isset(nodemask, i))
				return aml_binding_create(b, AML_BINDING_TYPE_SINGLE,i);
	}
	else if(data->policy == MPOL_INTERLEAVE)
	{
		return aml_binding_create(b, AML_BINDING_TYPE_INTERLEAVE,
					  nodemask);
	}
	return 0;
}


int aml_area_linux_mbind_regular_pre_bind(struct aml_area_linux_mbind_data *data)
{
	assert(data != NULL);
	return 0;
}

int aml_area_linux_mbind_regular_post_bind(struct aml_area_linux_mbind_data *data,
					   void *ptr, size_t sz)
{
	assert(data != NULL);
	/* realign ptr to match mbind requirement that it is aligned on a page */
	intptr_t aligned = (intptr_t)ptr & (intptr_t)(~(PAGE_SIZE -1));
	size_t end = sz + ((intptr_t)ptr - aligned);
	return mbind((void*)aligned, sz, data->policy, data->nodemask.mask, AML_BITMAP_MAX, 0);
}

struct aml_area_linux_mbind_ops aml_area_linux_mbind_regular_ops = {
	aml_area_linux_mbind_regular_pre_bind,
	aml_area_linux_mbind_regular_post_bind,
	aml_area_linux_mbind_generic_binding,
};

int aml_area_linux_mbind_setdata(struct aml_area_linux_mbind_data *data,
				 int policy, const struct aml_bitmap *nodemask)
{
	assert(data != NULL);
	data->policy = policy;
	aml_bitmap_copy(&data->nodemask, nodemask);
	return 0;
}

int aml_area_linux_mbind_mempolicy_pre_bind(struct aml_area_linux_mbind_data *data)
{
	assert(data != NULL);
	/* function is called before mmap, we save the "generic" mempolicy into
	 * our data, and apply the one the user actually want
	 */
	int policy;
	int err;
	struct aml_bitmap bitmap;
	get_mempolicy(&policy, (unsigned long *)&bitmap.mask,
		      AML_BITMAP_MAX, NULL, 0);
	err = set_mempolicy(data->policy, (unsigned long *)&data->nodemask.mask,
			    AML_BITMAP_MAX);
	aml_area_linux_mbind_setdata(data, policy, &bitmap);
	return err;
}

int aml_area_linux_mbind_mempolicy_post_bind(struct aml_area_linux_mbind_data *data,
					     void *ptr, size_t sz)
{
	assert(data != NULL);
	/* function is called after mmap, we retrieve the mempolicy we applied
	 * to it, and restore the generic mempolicy we saved earlier.
	 */
	int policy;
	int err;
	struct aml_bitmap bitmap;
	get_mempolicy(&policy, (unsigned long *)&bitmap.mask,
		      AML_BITMAP_MAX, NULL, 0);
	err = set_mempolicy(data->policy, (unsigned long *)data->nodemask.mask,
			    AML_BITMAP_MAX);
	aml_area_linux_mbind_setdata(data, policy, &bitmap);
	return err;
}

struct aml_area_linux_mbind_ops aml_area_linux_mbind_mempolicy_ops = {
	aml_area_linux_mbind_mempolicy_pre_bind,
	aml_area_linux_mbind_mempolicy_post_bind,
	aml_area_linux_mbind_generic_binding,
};

int aml_area_linux_mbind_init(struct aml_area_linux_mbind_data *data,
			      int policy, const struct aml_bitmap *nodemask)
{
	assert(data != NULL);
	aml_area_linux_mbind_setdata(data, policy, nodemask);
	return 0;
}

int aml_area_linux_mbind_destroy(struct aml_area_linux_mbind_data *data)
{
	assert(data != NULL);
	return 0;
}