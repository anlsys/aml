/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include "aml.h"

#include "aml/higher/replicaset.h"

int aml_replicaset_init(struct aml_replicaset *replicaset, const void *data)
{
	if (replicaset == NULL || data == NULL)
		return -AML_EINVAL;
	if (replicaset->ops->init == NULL)
		return -AML_ENOTSUP;
	return replicaset->ops->init(replicaset, data);
}

int aml_replicaset_sync(struct aml_replicaset *replicaset,
                        const unsigned int id)
{
	if (replicaset == NULL)
		return -AML_EINVAL;
	if (replicaset->ops->sync == NULL)
		return -AML_ENOTSUP;
	return replicaset->ops->sync(replicaset, id);
}
