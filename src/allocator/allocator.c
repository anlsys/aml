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

int is_power_of_two(size_t size)
{
	char i = 1;
	const char nbits = sizeof(size_t) * 8;

	if (size == 0)
		return 0;

	// Shift right until we meet a bit that is not 0.
	while (((size >> i) << i) == size)
		i++;
	// Shift left by (nbits-i) bits. If size is the same,
	// then only one bit is set and size is a power of two.
	i = nbits - i;
	return ((size << i) >> i) == size;
}

void *aml_aligned_alloc(struct aml_allocator *allocator,
                        size_t size,
                        size_t alignement)
{
	if (allocator == NULL || allocator->data == NULL ||
	    allocator->ops == NULL || allocator->ops->alloc == NULL ||
	    !is_power_of_two(alignement) || size == 0) {
		aml_errno = AML_EINVAL;
		return NULL;
	}

	if (allocator->ops->aligned_alloc != NULL)
		return allocator->ops->aligned_alloc(allocator->data, size,
		                                     alignement);
	else
		return allocator->ops->alloc(allocator->data, size);
}

int aml_free(struct aml_allocator *allocator, void *ptr)
{
	if (allocator == NULL || allocator->data == NULL ||
	    allocator->ops == NULL || allocator->ops->free == NULL)
		return -AML_EINVAL;

	return allocator->ops->free(allocator->data, ptr);
}
