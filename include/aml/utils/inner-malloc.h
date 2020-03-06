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

#include "aml/utils/macros.h"

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

/** Returns the size required for allocation of up to 8 types **/
#define AML_SIZEOF_ALIGNED(...) sizeof(AML_STRUCT_DECL(__VA_ARGS__))

/**
 * Returns the size required for allocation of up to 7 types plus one array.
 * @param n: The number of elements in array.
 * @param type: The type of array elements.
 * @param ...: Up to 7 fields type preceding array allocation space.
 **/
#define AML_SIZEOF_ALIGNED_ARRAY(n, type, ...)				\
	(sizeof(AML_STRUCT_DECL(__VA_ARGS__, type)) +			\
	 ((n)-1) * sizeof(type))

/** Returns the offset of the nth type of a list of up to 8 types. **/
#define AML_OFFSETOF_ALIGNED(N, ...) \
	offsetof(AML_STRUCT_DECL(__VA_ARGS__), AML_FIELD(N))

//---------------------------------------------------------------------------//
// User Macros
//---------------------------------------------------------------------------//

/**
 * Allocate space aligned on a page boundary for up to 8 fields aligned as
 * in a struct
 * @param ...: types contained in allocation. (Up to 8)
 **/
#define AML_INNER_MALLOC(...) calloc(1, AML_SIZEOF_ALIGNED(__VA_ARGS__))

/**
 * Allocate space aligned on a page boundary. It may contain up to 7 fields
 * aligned as in a struct, and one array.
 * @param n: Number of elements in array.
 * @param type: Type of array elements.
 * @param ...: Up to 7 fields type preceding array allocation space.
 **/
#define AML_INNER_MALLOC_ARRAY(n, type, ...) \
	calloc(1, AML_SIZEOF_ALIGNED_ARRAY(n, type, __VA_ARGS__))

/**
 * Allocate space aligned on a page boundary. It may contain up to 7 fields
 * aligned as in a struct, one aligned array and arbitrary extra space.
 * @param n: Number of elements in array.
 * @param type: Type of array elements.
 * @param size: The extra space in bytes to allocate.
 * @param ...: Up to 7 fields type preceding array allocation space.
 **/
#define AML_INNER_MALLOC_EXTRA(n, type, size, ...) \
	calloc(1, AML_SIZEOF_ALIGNED_ARRAY(n, type, __VA_ARGS__) + size)

/**
 * Returns the nth __VA__ARGS__ field pointer from AML_INNER_MALLOC*()
 * allocation.
 * @param ptr: A pointer obtained from AML_INNER_MALLOC*()
 * @param N: The field number. N must be a number (1, 2, 3, 4, 5, 6, 7, 8)
 * and not a variable.
 * @param ...: types contained in allocation. (Up to 8)
 * @return A pointer to Nth field after ptr.
 **/
#define AML_INNER_MALLOC_GET_FIELD(ptr, N, ...) \
	(void *)(((intptr_t) ptr) + AML_OFFSETOF_ALIGNED(N, __VA_ARGS__))

/**
 * Returns a pointer to the array after __VA_ARGS__ fields.
 * @param ptr: Pointer returned by AML_INNER_MALLOC_ARRAY() or
 * AML_INNER_MALLOC_EXTRA().
 * @param type: Type of array elements.
 * @param ...: Other types contained in allocation. (Up to 7)
 **/
#define AML_INNER_MALLOC_GET_ARRAY(ptr, type, ...)			\
	AML_INNER_MALLOC_GET_FIELD(ptr,					\
				   PLUS_1(VA_NARG(__VA_ARGS__)),	\
				   __VA_ARGS__, type)

/**
 * Returns a pointer to extra space allocated with
 * AML_INNER_MALLOC_EXTRA().
 * @param ptr: Pointer returned by AML_INNER_MALLOC_EXTRA().
 * @param n: Number of elements in the array.
 * @param type: Type of elements in the array.
 * @param ...: Other types contained in allocation. (Up to 7)
 **/
#define AML_INNER_MALLOC_GET_EXTRA(ptr, n, type, ...)			\
	(void *)(((intptr_t) ptr) +					\
		 AML_SIZEOF_ALIGNED_ARRAY(n, type, __VA_ARGS__))

/**
 * @}
 **/

#endif //AML_INNER_MALLOC_H
