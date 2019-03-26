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
