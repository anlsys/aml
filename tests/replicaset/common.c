/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include "test_replicaset.h"

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
