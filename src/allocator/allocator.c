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

# include <assert.h>

void *aml_alloc(struct aml_allocator *allocator, size_t size)
{
    if (allocator == NULL || allocator->data == NULL || allocator->ops == NULL) {
        aml_errno = AML_EINVAL;
        return NULL;
    }
    assert(allocator->ops->alloc);
    return allocator->ops->alloc(allocator->data, size);
}

struct aml_allocator_chunk *aml_allocator_alloc_chunk(struct aml_allocator *allocator, size_t size)
{
    if (allocator == NULL || allocator->data == NULL || allocator->ops == NULL) {
        aml_errno = AML_EINVAL;
        return NULL;
    }
    if (allocator->ops->alloc_chunk == NULL) {
        aml_errno = AML_ENOTSUP;
        return NULL;
    }
    return allocator->ops->alloc_chunk(allocator->data, size);
}

int aml_free(struct aml_allocator *allocator, void *ptr)
{
    if (ptr == NULL)
        return AML_SUCCESS;
    if (allocator == NULL || allocator->data == NULL || allocator->ops == NULL)
        return -AML_EINVAL;
    assert(allocator->ops->free);
    return allocator->ops->free(allocator->data, ptr);
}

int aml_allocator_free_chunk(struct aml_allocator *allocator, struct aml_allocator_chunk *chunk)
{
    if (chunk == NULL)
        return AML_SUCCESS;
    if (allocator == NULL || allocator->data == NULL || allocator->ops == NULL)
        return -AML_EINVAL;
    if (allocator->ops->free_chunk == NULL)
        return -AML_ENOTSUP;
    return allocator->ops->free_chunk(allocator->data, chunk);
}
