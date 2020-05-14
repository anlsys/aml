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

#include "aml/area/hwloc.h"

extern hwloc_topology_t aml_topology;

/** Number of sizes to test **/
#define ns 3
size_t sizes[ns] = {
        256 /* Less than a page */, 5000 /* Not a multiple of page size */,
        (1 << 20) /* A "large" size */
};

void check_area_size(struct aml_area *area, const size_t size)
{
	void *ptr = NULL;

	ptr = aml_area_mmap(area, size, NULL);
	assert(ptr != NULL);
	assert(aml_area_hwloc_check_binding(area, ptr, size));
	aml_area_munmap(area, ptr, size);
}

void check_nodeset(hwloc_bitmap_t nodeset,
                   const int np,
                   const hwloc_membind_policy_t *policies)
{
	int s, p, err;
	hwloc_membind_policy_t policy;
	struct aml_area *area;

	for (p = 0; p < np; p++) {
		policy = policies[p];
		err = aml_area_hwloc_create(&area, nodeset, policy);
		if (err != -AML_ENOTSUP) {
			assert(err == AML_SUCCESS);
			for (s = 0; s < ns; s++)
				check_area_size(area, sizes[s]);
			aml_area_hwloc_destroy(&area);
		}
	}
}

void check_areas()
{
	hwloc_bitmap_t nodeset;
	hwloc_obj_t node;

	nodeset = hwloc_bitmap_alloc();
	assert(nodeset != NULL);

	/* Test full topology nodeset */
	const hwloc_membind_policy_t sys_policies[5] = {
	        HWLOC_MEMBIND_DEFAULT, HWLOC_MEMBIND_FIRSTTOUCH,
	        HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_INTERLEAVE,
	        HWLOC_MEMBIND_NEXTTOUCH};
	node = hwloc_get_obj_by_type(aml_topology, HWLOC_OBJ_MACHINE, 0);
	assert(node != NULL);
	assert(node->nodeset != NULL);
	assert(hwloc_bitmap_weight(node->nodeset) > 0);
	check_nodeset(node->nodeset, 5, sys_policies);

	/* Test individual nodes */
	const hwloc_membind_policy_t local_policies[2] = {
	        HWLOC_MEMBIND_BIND,
	        HWLOC_MEMBIND_INTERLEAVE,
	};
	node = hwloc_get_obj_by_type(aml_topology, HWLOC_OBJ_NUMANODE, 0);
	while (node != NULL) {
		assert(node->nodeset != NULL);
		assert(hwloc_bitmap_weight(node->nodeset) > 0);
		check_nodeset(node->nodeset, 2, local_policies);
		node = node->next_cousin;
	}

	/* Test odd nodes */
	node = NULL;
	while ((node = hwloc_get_next_obj_by_type(
	                aml_topology, HWLOC_OBJ_NUMANODE, node)) != NULL) {
		assert(node->nodeset != NULL);
		assert(hwloc_bitmap_weight(node->nodeset) > 0);
		hwloc_bitmap_or(nodeset, nodeset, node->nodeset);
		node = hwloc_get_next_obj_by_type(aml_topology,
		                                  HWLOC_OBJ_NUMANODE, node);
		if (node == NULL)
			break;
	}
	check_nodeset(nodeset, 2, local_policies);

	/* Test even nodes */
	hwloc_bitmap_zero(nodeset);
	node = NULL;
	while ((node = hwloc_get_next_obj_by_type(
	                aml_topology, HWLOC_OBJ_NUMANODE, node)) != NULL) {
		node = hwloc_get_next_obj_by_type(aml_topology,
		                                  HWLOC_OBJ_NUMANODE, node);
		if (node == NULL)
			break;
		assert(node->nodeset != NULL);
		assert(hwloc_bitmap_weight(node->nodeset) > 0);
		hwloc_bitmap_or(nodeset, nodeset, node->nodeset);
	}
	if (hwloc_bitmap_weight(nodeset) > 0)
		check_nodeset(nodeset, 2, local_policies);

	hwloc_bitmap_free(nodeset);
}

int main(int argc, char **argv)
{
	aml_init(&argc, &argv);
	check_areas();
	aml_finalize();
}
