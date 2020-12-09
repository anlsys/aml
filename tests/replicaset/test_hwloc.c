/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include "test_replicaset.h"

#include <stdlib.h>

#include "aml/higher/replicaset/hwloc.h"

const size_t size = 1 << 20; // 1 Mo
void *src;

void test_case(const hwloc_obj_type_t initiator_type,
               const enum hwloc_distances_kind_e kind)
{
	struct aml_replicaset *replicaset;
	assert(aml_replicaset_hwloc_create(&replicaset, size, initiator_type,
	                                   kind) == AML_SUCCESS);
	test_replicaset(replicaset, src, memcmp);
	aml_replicaset_hwloc_destroy(&replicaset);
}

int main(int argc, char **argv)
{
	// Setup
	aml_init(&argc, &argv);
	src = malloc(size);
	assert(src);
	memset(src, '#', size);

	test_case(HWLOC_OBJ_PU, HWLOC_DISTANCES_KIND_MEANS_LATENCY);
	test_case(HWLOC_OBJ_CORE, HWLOC_DISTANCES_KIND_MEANS_BANDWIDTH);
	test_case(HWLOC_OBJ_NUMANODE, HWLOC_DISTANCES_KIND_MEANS_LATENCY);

	// Teardown
	free(src);
	aml_finalize();
	return 0;
}
