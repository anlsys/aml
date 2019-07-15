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

#define AML_AREA_LINUX_MBIND_FLAGS MPOL_MF_MOVE

int
aml_area_linux_mbind(struct aml_area_linux_data    *bind,
		     void                          *ptr,
		     size_t                         size)
{
	struct bitmask *nodeset;

	assert(bind != NULL);

	nodeset = bind->nodeset;
	if (nodeset == NULL)
		nodeset = numa_all_nodes_ptr;


	long err = mbind(ptr,
			 size,
			 bind->binding_flags,
			 nodeset->maskp,
			 nodeset->size,
			 AML_AREA_LINUX_MBIND_FLAGS);

	if (err == 0)
		return AML_SUCCESS;
	return -AML_FAILURE;
}

int
aml_area_linux_check_binding(struct aml_area_linux_data *area_data,
			     void                       *ptr,
			     size_t                      size)
{
	int err, mode, i;
	struct bitmask *nodeset;
	// unused parameter
	(void)size;

	nodeset = numa_allocate_nodemask();
	if (nodeset == NULL)
		return -AML_ENOMEM;

	err = get_mempolicy(&mode,
			    nodeset->maskp,
			    nodeset->size,
			    ptr,
			    AML_AREA_LINUX_MBIND_FLAGS);

	if (err < 0) {
		err = -AML_EINVAL;
		goto out;
	}

	err = 1;
	if (mode != area_data->binding_flags)
		err = 0;
	for (i = 0; i < numa_max_possible_node(); i++) {
		int ptr_set = numa_bitmask_isbitset(nodeset, i);
		int bitmask_set = numa_bitmask_isbitset(area_data->nodeset, i);

		if (mode == AML_AREA_LINUX_BINDING_FLAG_BIND &&
		    ptr_set != bitmask_set)
			goto binding_failed;
		if (mode == AML_AREA_LINUX_BINDING_FLAG_INTERLEAVE &&
		    ptr_set && !bitmask_set)
			goto binding_failed;
	}

	goto out;
 binding_failed:
	err = 0;
 out:
	numa_free_nodemask(nodeset);
	return err;
}

void *aml_area_linux_mmap(const struct aml_area_data  *area_data,
			  void                        *ptr,
			  size_t                       size)
{
	struct aml_area_linux_data *data =
		(struct aml_area_linux_data *) area_data;

	void *out = mmap(ptr, size, PROT_READ|PROT_WRITE,
			 data->mmap_flags, 0, 0);

	if (out == MAP_FAILED) {
		out = NULL;
		aml_errno = AML_FAILURE;
	}

	return out;
}

int aml_area_linux_munmap(
		__attribute__ ((unused)) const struct aml_area_data *area_data,
		void *ptr, const size_t size)
{
	int err = munmap(ptr, size);

	if (err == -1)
		return -AML_FAILURE;
	return AML_SUCCESS;
}


void *aml_area_linux_mmap_mbind(const struct aml_area_data  *area_data,
				void                        *ptr,
				size_t                       size)
{
	void *out = aml_area_linux_mmap(area_data, ptr, size);

	if (out == NULL)
		return NULL;

	struct aml_area_linux_data *data =
		(struct aml_area_linux_data *) area_data;

	if (data->nodeset != NULL || data->binding_flags != MPOL_DEFAULT) {
		int err = aml_area_linux_mbind(data, out, size);

		if (err != AML_SUCCESS) {
			aml_errno = -err;
			munmap(out, size);
			return NULL;
		}
	}

	return out;
}


/*******************************************************************************
 * Areas Initialization
 ******************************************************************************/

static int aml_area_linux_check_mmap_flags(const int mmap_flags)
{
	switch (mmap_flags) {
	case AML_AREA_LINUX_MMAP_FLAG_PRIVATE:
		break;
	case AML_AREA_LINUX_MMAP_FLAG_SHARED:
		break;
	default:
		return 0;
	}

	return 1;
}

static int aml_area_linux_check_binding_flags(const int binding_flags)
{
	switch (binding_flags) {
	case AML_AREA_LINUX_BINDING_FLAG_BIND:
		break;
	case AML_AREA_LINUX_BINDING_FLAG_INTERLEAVE:
		break;
	case AML_AREA_LINUX_BINDING_FLAG_PREFERRED:
		break;
	default:
		return 0;
	}

	return 1;
}

int aml_area_linux_create(struct aml_area **area, const int mmap_flags,
			  const struct aml_bitmap *nodemask,
			  const int binding_flags)
{
	struct aml_area *ret = NULL;
	struct aml_area_linux_data *data;
	int err;

	if (area == NULL)
		return -AML_EINVAL;

	*area = NULL;

	/* check flags */
	if (!aml_area_linux_check_mmap_flags(mmap_flags) ||
	    !aml_area_linux_check_binding_flags(binding_flags)) {
		return -AML_EINVAL;
	}

	ret = calloc(1, sizeof(struct aml_area));
	if (ret == NULL)
		return -AML_ENOMEM;

	ret->ops = &aml_area_linux_ops;
	ret->data = calloc(1, sizeof(struct aml_area_linux_data));
	if (ret->data == NULL) {
		err = -AML_ENOMEM;
		goto err_f_ret;
	}
	data = (struct aml_area_linux_data *)ret->data;

	/* set area_data and area */
	data->binding_flags = binding_flags;
	data->mmap_flags = mmap_flags;

	/* check/set nodemask */
	data->nodeset = numa_get_mems_allowed();
	if (data->nodeset == NULL) {
		err = -AML_ENOMEM;
		goto err_f_data;
	}

	/* check if the nodemask is compatible with the nodeset */
	if (nodemask != NULL) {
		int aml_last = aml_bitmap_last(nodemask);
		int allowed_last = numa_bitmask_weight(data->nodeset);

		while (!numa_bitmask_isbitset(data->nodeset, --allowed_last))
			;

		if (aml_last > allowed_last) {
			err = -AML_EDOM;
			goto err_f_node;
		}
		aml_bitmap_copy_to_ulong(nodemask,
					 data->nodeset->maskp,
					 data->nodeset->size);
	}
	*area = ret;
	return AML_SUCCESS;
err_f_node:
	numa_free_nodemask(data->nodeset);
err_f_data:
	free(ret->data);
err_f_ret:
	free(ret);
	return err;
}

void aml_area_linux_destroy(struct aml_area **area)
{
	struct aml_area *a;
	struct aml_area_linux_data *data;

	if (area == NULL)
		return;
	a = *area;
	if (a == NULL || a->data == NULL)
		return;

	data = (struct aml_area_linux_data *) a->data;
	numa_free_nodemask(data->nodeset);
	free(data);
	free(a);
	*area = NULL;
}


/*******************************************************************************
 * Areas declaration
 ******************************************************************************/

struct aml_area_linux_data aml_area_linux_data_default = {
	.nodeset = NULL,
	.binding_flags = MPOL_DEFAULT,
	.mmap_flags = AML_AREA_LINUX_MMAP_FLAG_PRIVATE
};

struct aml_area_ops aml_area_linux_ops = {
	.mmap = aml_area_linux_mmap_mbind,
	.munmap = aml_area_linux_munmap
};

struct aml_area aml_area_linux = {
	.ops = &aml_area_linux_ops,
	.data = (struct aml_area_data *)(&aml_area_linux_data_default)
};

