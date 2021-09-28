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

#include "aml/higher/allocator.h"

void *aml_alloc(struct aml_allocator *allocator, size_t size)
{
	if (allocator == NULL || allocator->data == NULL ||
	    allocator->ops == NULL || allocator->ops->alloc == NULL) {
		aml_errno = AML_EINVAL;
		return NULL;
	}

	return allocator->ops->alloc(allocator->data, size);
}

int aml_free(struct aml_allocator *allocator, void *ptr)
{
	if (allocator == NULL || allocator->data == NULL ||
	    allocator->ops == NULL || allocator->ops->free == NULL)
		return -AML_EINVAL;

	if (ptr == NULL)
		return AML_SUCCESS;

	return allocator->ops->free(allocator->data, ptr);
}
