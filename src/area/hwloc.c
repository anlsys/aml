/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include <sys/mman.h>

#include "aml.h"

#include "aml/area/hwloc.h"

#define HWLOC_BINDING_FLAGS                                                    \
	(HWLOC_MEMBIND_PROCESS | HWLOC_MEMBIND_NOCPUBIND |                     \
	 HWLOC_MEMBIND_BYNODESET | HWLOC_MEMBIND_STRICT)

/**
 * Topology used by AML.
 * Topology is initalized when calling aml_library_init().
 **/
extern hwloc_topology_t aml_topology;
extern hwloc_const_bitmap_t allowed_nodeset;

/**************************************************************************/
/* aml_area_hwloc                                                         */
/**************************************************************************/

static int aml_area_hwloc_init(struct aml_area *area,
                               hwloc_bitmap_t nodeset,
                               const hwloc_membind_policy_t policy)
{
	struct aml_area_hwloc_data *d =
	        (struct aml_area_hwloc_data *)area->data;

	// Check support
	const struct hwloc_topology_support *sup =
	        hwloc_topology_get_support(aml_topology);
	if (policy == HWLOC_MEMBIND_BIND && !sup->membind->bind_membind)
		return -AML_ENOTSUP;
	if (policy == HWLOC_MEMBIND_FIRSTTOUCH &&
	    !sup->membind->firsttouch_membind)
		return -AML_ENOTSUP;
	if (policy == HWLOC_MEMBIND_INTERLEAVE &&
	    !sup->membind->interleave_membind)
		return -AML_ENOTSUP;
	if (policy == HWLOC_MEMBIND_NEXTTOUCH &&
	    !sup->membind->nexttouch_membind)
		return -AML_ENOTSUP;

	// Check nodeset does not include unallowed nodes
	if (nodeset && !hwloc_bitmap_isincluded(nodeset, allowed_nodeset))
		return -AML_EDOM;

	// Set area nodeset and policy
	if (nodeset == NULL)
		d->nodeset = hwloc_bitmap_dup(allowed_nodeset);
	else
		d->nodeset = hwloc_bitmap_dup(nodeset);
	d->policy = policy;

	return AML_SUCCESS;
}

int aml_area_hwloc_create(struct aml_area **area,
                          hwloc_bitmap_t nodeset,
                          const hwloc_membind_policy_t policy)
{
	int err = AML_SUCCESS;
	struct aml_area *a;
	struct aml_area_hwloc_data *data;

	a = AML_INNER_MALLOC(struct aml_area, struct aml_area_hwloc_data);
	if (a == NULL)
		return -AML_ENOMEM;

	a->ops = &aml_area_hwloc_ops;
	data = AML_INNER_MALLOC_GET_FIELD(a, 2, struct aml_area,
	                                  struct aml_area_hwloc_data);
	a->data = (struct aml_area_data *)data;

	err = aml_area_hwloc_init(a, nodeset, policy);
	if (err != AML_SUCCESS) {
		free(a);
		return err;
	}
	*area = a;
	return AML_SUCCESS;
}

void aml_area_hwloc_destroy(struct aml_area **area)
{
	if (area == NULL || *area == NULL)
		return;

	struct aml_area_hwloc_data *data =
	        (struct aml_area_hwloc_data *)(*area)->data;

	hwloc_bitmap_free(data->nodeset);
	free(*area);
	*area = NULL;
}

void *aml_area_hwloc_mmap(const struct aml_area_data *data,
                          size_t size,
                          struct aml_area_mmap_options *opts)
{
	(void)opts;
	struct aml_area_hwloc_data *hwloc_data =
	        (struct aml_area_hwloc_data *)data;

	hwloc_const_bitmap_t nodeset = hwloc_data->nodeset == NULL ?
	                                       allowed_nodeset :
	                                       hwloc_data->nodeset;

	void *p = hwloc_alloc_membind(aml_topology, size, nodeset,
	                              hwloc_data->policy, HWLOC_BINDING_FLAGS);

	if (p == NULL) {
		if (errno == EINVAL)
			aml_errno = AML_EINVAL;
		if (errno == ENOSYS)
			aml_errno = AML_ENOTSUP;
		if (errno == EXDEV)
			aml_errno = AML_FAILURE;
		if (errno == ENOMEM)
			aml_errno = AML_ENOMEM;
		return NULL;
	}

	return p;
}

int aml_area_hwloc_munmap(const struct aml_area_data *data,
                          void *ptr,
                          size_t size)
{
	(void)data;
	hwloc_free(aml_topology, ptr, size);
	return AML_SUCCESS;
}

int aml_area_hwloc_check_binding(struct aml_area *area, void *ptr, size_t size)
{
	int err;
	hwloc_membind_policy_t policy;
	hwloc_bitmap_t nodeset;
	struct aml_area_hwloc_data *bind =
	        (struct aml_area_hwloc_data *)area->data;

	nodeset = hwloc_bitmap_alloc();

	if (nodeset == NULL)
		return -AML_ENOMEM;

	err = hwloc_get_area_membind(aml_topology, ptr, size, nodeset, &policy,
	                             HWLOC_BINDING_FLAGS);

	if (err < 0) {
		err = AML_EINVAL;
		goto out;
	}

	err = 1;

	if (policy != bind->policy && bind->policy != HWLOC_MEMBIND_DEFAULT &&
	    bind->policy != HWLOC_MEMBIND_MIXED)
		err = 0;

	if (!hwloc_bitmap_isequal(nodeset, bind->nodeset))
		err = 0;

out:
	hwloc_bitmap_free(nodeset);
	return err;
}

/**
 * Static declaration of an area.
 * No check is performed to enforced availability of policies and nodeset.
 * If bad values are provided area function call will return
 * an error.
 * @param[in] name: The area instance name.
 * @param[in] bitmap: The nodeset to set inside area (hwloc_bitmap_t).
 *        Can be NULL.
 * @param[in] bind_policy: The memory allocation policy
 *        (hwloc_membind_policy_t).
 **/
#define AML_AREA_HWLOC_DECL(name, bitmap, bind_policy)                         \
	struct aml_area_hwloc_data __##name##_data = {.nodeset = bitmap,       \
	                                              .policy = bind_policy};  \
	struct aml_area name = {                                               \
	        .ops = &aml_area_hwloc_ops,                                    \
	        .data = (struct aml_area_data *)&__##name##_data}

struct aml_area_ops aml_area_hwloc_ops = {
        .mmap = aml_area_hwloc_mmap,
        .munmap = aml_area_hwloc_munmap,
        .fprintf = NULL,
};

AML_AREA_HWLOC_DECL(aml_area_hwloc_default, NULL, HWLOC_MEMBIND_DEFAULT);

AML_AREA_HWLOC_DECL(aml_area_hwloc_interleave, NULL, HWLOC_MEMBIND_INTERLEAVE);

AML_AREA_HWLOC_DECL(aml_area_hwloc_firsttouch, NULL, HWLOC_MEMBIND_FIRSTTOUCH);
