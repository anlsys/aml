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
#include <inttypes.h>
#include <sys/mman.h>
#include <numa.h>
#include <numaif.h>
#include <assert.h>

#define AML_AREA_LINUX_MBIND_FLAGS MPOL_MF_MOVE

static inline int aml_area_linux_policy(const enum aml_area_linux_policy p)
{
	switch (p) {
	case AML_AREA_LINUX_POLICY_DEFAULT:
		return MPOL_DEFAULT;
	case AML_AREA_LINUX_POLICY_BIND:
		return MPOL_BIND;
	case AML_AREA_LINUX_POLICY_PREFERRED:
		return MPOL_PREFERRED;
	case AML_AREA_LINUX_POLICY_INTERLEAVE:
		return MPOL_INTERLEAVE;
	}
	return MPOL_DEFAULT;
}

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
			 aml_area_linux_policy(bind->policy),
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
	if (mode != aml_area_linux_policy(area_data->policy))
		err = 0;
	for (i = 0; i < numa_max_possible_node(); i++) {
		int ptr_set = numa_bitmask_isbitset(nodeset, i);
		int bitmask_set = numa_bitmask_isbitset(area_data->nodeset, i);

		if (mode == AML_AREA_LINUX_POLICY_BIND &&
		    ptr_set != bitmask_set)
			goto binding_failed;
		if (mode == AML_AREA_LINUX_POLICY_INTERLEAVE &&
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

void *aml_area_linux_mmap(const struct aml_area_data   *area_data,
			  size_t                        size,
			  struct aml_area_mmap_options *opts)
{
	(void) area_data;
	void *out;
	struct aml_area_linux_mmap_options *options;

	options = (struct aml_area_linux_mmap_options *) opts;
	if (options == NULL) {
		out = mmap(NULL,
			   size,
			   PROT_READ|PROT_WRITE,
			   MAP_PRIVATE | MAP_ANONYMOUS,
			   0,
			   0);
	} else {
		out = mmap(options->ptr,
			   size,
			   options->mode,
			   options->flags,
			   options->fd,
			   options->offset);
	}

	if (out == MAP_FAILED) {
		out = NULL;
		aml_errno = AML_FAILURE;
	}

	return out;
}

int aml_area_linux_munmap(const struct aml_area_data *area_data,
		void *ptr, const size_t size)
{
	(void)area_data;
	int err = munmap(ptr, size);

	if (err == -1)
		return -AML_FAILURE;
	return AML_SUCCESS;
}


void *aml_area_linux_mmap_mbind(const struct aml_area_data  *area_data,
				size_t                       size,
				struct aml_area_mmap_options *opts)
{
	void *out = aml_area_linux_mmap(area_data, size, opts);

	if (out == NULL)
		return NULL;

	struct aml_area_linux_data *data =
		(struct aml_area_linux_data *) area_data;

	if (data->nodeset != NULL ||
	    data->policy != AML_AREA_LINUX_POLICY_DEFAULT) {
		int err = aml_area_linux_mbind(data, out, size);

		if (err != AML_SUCCESS) {
			aml_errno = -err;
			munmap(out, size);
			return NULL;
		}
	}

	return out;
}

int aml_area_linux_fprintf(const struct aml_area_data *data,
			   FILE *stream, const char *prefix)
{
	const struct aml_area_linux_data *d;

	static const char * const policies[] = {
		[AML_AREA_LINUX_POLICY_DEFAULT] = "default",
		[AML_AREA_LINUX_POLICY_BIND] = "bind",
		[AML_AREA_LINUX_POLICY_PREFERRED] = "preferred",
		[AML_AREA_LINUX_POLICY_INTERLEAVE] = "interleave"
	};

	fprintf(stream, "%s: area-linux: %p\n", prefix, (void *)data);
	if (data == NULL)
		return AML_SUCCESS;

	d = (const struct aml_area_linux_data *)data;

	if (d->nodeset == NULL)
		fprintf(stream, "%s: bitmask: 0x0\n", prefix);
	else {
		fprintf(stream, "%s: bitmask: size: %zu", prefix,
			d->nodeset->size);
		fprintf(stream, "%s: bitmask: maskp: ", prefix);
		for (size_t i = 0; i < d->nodeset->size; i++)
			fprintf(stream, "%8lx", d->nodeset->maskp[i]);
		fprintf(stream, "\n");
	}
	fprintf(stream, "%s: policy: %s\n", prefix, policies[d->policy]);
	return AML_SUCCESS;
}

/*******************************************************************************
 * Areas Initialization
 ******************************************************************************/

int aml_area_linux_create(struct aml_area **area,
			  const struct aml_bitmap *nodemask,
			  const enum aml_area_linux_policy policy)
{
	struct aml_area *ret = NULL;
	struct aml_area_linux_data *data;
	int err;

	if (area == NULL)
		return -AML_EINVAL;

	*area = NULL;

	ret = AML_INNER_MALLOC(struct aml_area,
				      struct aml_area_linux_data);
	if (ret == NULL)
		return -AML_ENOMEM;

	ret->data = AML_INNER_MALLOC_GET_FIELD(ret, 2,
					       struct aml_area,
					       struct aml_area_linux_data);
	ret->ops = &aml_area_linux_ops;
	data = (struct aml_area_linux_data *)ret->data;

	/* set area_data */
	data->policy = policy;

	/* check/set nodemask */
	data->nodeset = numa_get_mems_allowed();
	if (data->nodeset == NULL) {
		err = -AML_ENOMEM;
		goto err_f_ret;
	}

	/* check if the nodemask is compatible with the nodeset */
	if (nodemask != NULL) {
		for (int i = 0; i < AML_BITMAP_MAX; i++) {
			int ours, theirs;

			ours = aml_bitmap_isset(nodemask, i);
			theirs = numa_bitmask_isbitset(data->nodeset, i);

			if (ours && !theirs) {
				err = -AML_EDOM;
				goto err_f_node;
			}
		}
		aml_bitmap_copy_to_ulong(nodemask,
					 data->nodeset->maskp,
					 data->nodeset->size);
	}
	*area = ret;
	return AML_SUCCESS;
err_f_node:
	numa_free_nodemask(data->nodeset);
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
	if (a == NULL)
		return;

	/* with our creators it should not happen */
	assert(a->data != NULL);
	data = (struct aml_area_linux_data *) a->data;
	numa_free_nodemask(data->nodeset);
	free(a);
	*area = NULL;
}


/*******************************************************************************
 * Areas declaration
 ******************************************************************************/

struct aml_area_linux_data aml_area_linux_data_default = {
	.nodeset = NULL,
	.policy = AML_AREA_LINUX_POLICY_DEFAULT,
};

struct aml_area_ops aml_area_linux_ops = {
	.mmap = aml_area_linux_mmap_mbind,
	.munmap = aml_area_linux_munmap,
	.fprintf = aml_area_linux_fprintf,
};

struct aml_area aml_area_linux = {
	.ops = &aml_area_linux_ops,
	.data = (struct aml_area_data *)(&aml_area_linux_data_default)
};
