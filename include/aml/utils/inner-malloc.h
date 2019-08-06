/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
*******************************************************************************/

#ifndef AML_INNER_MALLOC_H
#define AML_INNER_MALLOC_H

/**
 * @defgroup aml_inner_malloc "AML Internal Allocation Management"
 * @brief AML helper functions to handle inner allocations
 * @{
 *
 * Set of macros to create properly sized allocations of our internal complex
 * objects. In particular, help with the generic handle and implementation
 * specific data allocation as a single allocation, with all pointer properly
 * aligned.
 *
 * This code is all macros to handle the type specific logic we need.
 **/

/** Returns the allocation size required to handle two objects side-by-side.
 *
 * Use an anonymous struct to ask the compiler what size an allocation should be
 * so that the second object is properly aligned too.
 */
#define AML_SIZEOF_ALIGNED(a, b) \
	(sizeof(struct { a __e1; b __e2; }))

/** Returns the offset of the second object when allocated side-by-side.
 *
 * Use the same anonymous struct trick to figure out what offset the pointer is
 * at.
 */
#define AML_OFFSETOF_ALIGNED(a, b) \
	(offsetof(struct { a __e1; b __e2; }, __e2))

/** Allocate a pointer that can be used to contain two types.
 *
 **/
#define AML_INNER_MALLOC_2(a, b) calloc(1, AML_SIZEOF_ALIGNED(a, b))

/** Allocate a pointer that can be used to contain two types plus an extra area
 * aligned on a third type.
 *
 **/
#define AML_INNER_MALLOC_EXTRA(a, b, c, sz) \
	calloc(1, AML_SIZEOF_ALIGNED(struct { a  __f1; b __f2; }, c) + \
	       (sizeof(c)*sz))

/** Returns the next pointer after an AML_INNER_MALLOC.
 *
 * Can be used to iterate over the pointers we need, using the last two types as
 * parameters.
 **/
#define AML_INNER_MALLOC_NEXTPTR(ptr, a, b) \
	(void *)(((intptr_t) ptr) + AML_OFFSETOF_ALIGNED(a, b))

/** Returns a pointer inside the extra zone after an AML_INNER_MALLOC_EXTRA.
 *
 * Can be used to iterate over the pointers we need.
 **/
#define AML_INNER_MALLOC_EXTRA_NEXTPTR(ptr, a, b, c, off) \
	(void *)(((intptr_t) ptr) + \
		 AML_OFFSETOF_ALIGNED(struct { a  __f1; b __f2; }, c) + \
		 ((off)*sizeof(c)))

/**
 * @}
 **/

#endif //AML_INNER_MALLOC_H
