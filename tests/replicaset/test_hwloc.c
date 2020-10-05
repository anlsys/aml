/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include <stdlib.h>

#include "aml.h"

#include "aml/higher/replicaset.h"
#include "aml/higher/replicaset/hwloc.h"

void test_replicaset(struct aml_replicaset *replicaset,
                     const void *src,
                     int (*comp)(const void *, const void *, size_t))
{
	// Basic tests
	assert(replicaset != NULL);
	assert(replicaset->ops != NULL);
	assert(replicaset->ops->init != NULL);
	assert(replicaset->ops->sync != NULL);
	assert(replicaset->n > 0);
	for (unsigned i = 0; i < replicaset->n; i++)
		assert(replicaset->replica[i] != NULL);

	// Check sync
	assert(aml_replicaset_sync(replicaset, 0) == AML_SUCCESS);
	for (unsigned i = 1; i < replicaset->n; i++)
		assert(!comp(replicaset->replica[i], replicaset->replica[0],
		             replicaset->size));

	// Check init
	assert(aml_replicaset_init(replicaset, src) == AML_SUCCESS);
	for (unsigned i = 0; i < replicaset->n; i++)
		assert(!comp(replicaset->replica[i], src, replicaset->size));
}

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
