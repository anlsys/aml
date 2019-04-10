/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
*******************************************************************************/

#include "aml.h"
#include <stdlib.h>

void *aml_area_mmap(const struct aml_area *area, void **ptr, size_t size)
{
	if (size == 0)
		return NULL;

	if (area == NULL) {
		aml_errno = AML_EINVAL;
		return NULL;
	}

	if (area->ops->mmap == NULL) {
		aml_errno = AML_ENOTSUP;
		return NULL;
	}


	return area->ops->mmap(area->data, ptr, size);
}

int aml_area_munmap(const struct aml_area *area, void *ptr, size_t size)
{
	if (ptr == NULL || size == 0)
		return AML_SUCCESS;

	if (area == NULL)
		return -AML_EINVAL;

	if (area->ops->munmap == NULL)
		return -AML_ENOTSUP;

	return area->ops->munmap(area->data, ptr, size);
}

