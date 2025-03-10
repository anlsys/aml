/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *******************************************************************************/
#include "config.h"

#include <assert.h>
#include <hwloc.h>
#include <stdlib.h>

#include "aml.h"

#include "aml/area/hwloc.h"

extern hwloc_topology_t aml_topology;

const char *xml_topology_path = "aml_topology.xml";

int aml_hwloc_distance_hop_matrix(const hwloc_obj_type_t ta,
                                  const hwloc_obj_type_t tb,
                                  struct hwloc_distances_s **s);

//------------------------------------------------------------------------------
// Test basic API
//------------------------------------------------------------------------------

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

//------------------------------------------------------------------------------
// Test preferred API
//------------------------------------------------------------------------------

void create_topology()
{
	unsigned nr = 1;
	struct hwloc_distances_s *hops, *xml_hops;

	// Get distance matrix
	assert(aml_hwloc_distance_hop_matrix(HWLOC_OBJ_CORE, HWLOC_OBJ_CORE,
	                                     &hops) == 0);

	// Add matrix to topology
	hwloc_distances_add_handle_t handle;
	handle = hwloc_distances_add_create(aml_topology, NULL, hops->kind, 0);
	assert(handle);
	assert(hwloc_distances_add_values(aml_topology, handle, hops->nbobjs,
	                                  hops->objs, hops->values, 0) == 0);
	assert(hwloc_distances_add_commit(
	               aml_topology, handle,
	               HWLOC_DISTANCES_ADD_FLAG_GROUP_INACCURATE) == 0);

	// Save topology as a xml file
	assert(hwloc_topology_export_xml(aml_topology, xml_topology_path, 0) !=
	       -1);

	aml_finalize();

	// Load xml topology.
	setenv("AML_TOPOLOGY", xml_topology_path, 1);
	assert(aml_init(NULL, NULL) == AML_SUCCESS);

	// Get xml distance matrix
	assert(hwloc_distances_get(aml_topology, &nr, &xml_hops,
	                           HWLOC_DISTANCES_KIND_FROM_USER |
	                                   HWLOC_DISTANCES_KIND_MEANS_LATENCY,
	                           0) == 0);

	// At least one distance matrix is present
	assert(nr > 0);
	// Same number of objects
	assert(hops->nbobjs == xml_hops->nbobjs);
	// Same values
	assert(!memcmp(hops->values, xml_hops->values,
	               hops->nbobjs * sizeof(*hops->values)));

	hwloc_distances_release(aml_topology, xml_hops);
	free(hops);
}

void test_preferred()
{
	// bind ourselves to the last NUMANODE.
	hwloc_obj_t PU, NUMANODE = hwloc_get_obj_by_type(aml_topology,
	                                                 HWLOC_OBJ_NUMANODE, 0);

	while (NUMANODE->next_cousin != NULL)
		NUMANODE = NUMANODE->next_cousin;

	PU = NUMANODE->parent;
	while (PU->type != HWLOC_OBJ_CORE)
		PU = PU->first_child;

	assert(hwloc_set_cpubind(aml_topology, PU->cpuset,
	                         HWLOC_CPUBIND_PROCESS | HWLOC_CPUBIND_STRICT |
	                                 HWLOC_CPUBIND_NOMEMBIND) != -1);

	// Now hwloc preferred area should return the
	// closest NUMANODE (i.e the one above)
	struct aml_area *area;
	struct aml_area_hwloc_preferred_data *data;

	assert(aml_area_hwloc_preferred_local_create(
	               &area, HWLOC_DISTANCES_KIND_FROM_USER |
	                              HWLOC_DISTANCES_KIND_MEANS_LATENCY) ==
	       AML_SUCCESS);

	data = (struct aml_area_hwloc_preferred_data *)area->data;
	assert(data->numanodes[0] == NUMANODE);

	aml_area_hwloc_preferred_destroy(&area);
}

void delete_topology()
{
	// Cleanup
	unlink(xml_topology_path);
}

int main(int argc, char **argv)
{
	/* impossible to do those check in a CI environment consistently */
	if (!strcmp(getenv("CI"), "true"))
		exit(77);

	aml_init(&argc, &argv);
	check_areas();
	create_topology();
	test_preferred();
	delete_topology();
	aml_finalize();
}
