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

# define IMPL_PTR(NAME, ...)                                                    \
    do {                                                                        \
        if (allocator == NULL || allocator->data == NULL ||                     \
                allocator->ops == NULL || allocator->ops->NAME == NULL) {       \
            aml_errno = AML_EINVAL;                                             \
            return NULL;                                                        \
        }                                                                       \
        return allocator->ops->NAME(__VA_ARGS__);                               \
    } while (0)

# define IMPL_INT(NAME,...)                                                     \
    do {                                                                        \
        if (allocator == NULL || allocator->data == NULL ||                     \
                allocator->ops == NULL || allocator->ops->NAME == NULL)         \
            return -AML_EINVAL;                                                 \
        return allocator->ops->NAME(__VA_ARGS__);                               \
    } while (0)

void *aml_alloc(struct aml_allocator *allocator, size_t size)
{
    IMPL_PTR(alloc, allocator->data, size);
}

struct aml_allocator_chunk *aml_allocator_alloc_chunk(struct aml_allocator *allocator, size_t size)
{
    IMPL_PTR(alloc_chunk, allocator->data, size);
}

int aml_free(struct aml_allocator *allocator, void *ptr)
{
    if (ptr == NULL)
        return AML_SUCCESS;
    IMPL_INT(free, allocator->data, ptr);
}

int aml_allocator_free_chunk(struct aml_allocator *allocator, struct aml_allocator_chunk *chunk)
{
    if (chunk == NULL)
        return AML_SUCCESS;
    IMPL_INT(free_chunk, allocator->data, chunk);
}
