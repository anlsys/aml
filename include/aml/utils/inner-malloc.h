/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#ifndef AML_INNER_MALLOC_H
#define AML_INNER_MALLOC_H

#ifdef __cplusplus
extern "C" {
#endif

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

//---------------------------------------------------------------------------//
// Inner utils
//---------------------------------------------------------------------------//

// Stringify macro
#define STRINGIFY(a) STRINGIFY_(a)
#define STRINGIFY_(a) #a

// Concatenate two arguments into a macro name
#define CONCATENATE(arg1, arg2)   CONCATENATE1(arg1, arg2)
#define CONCATENATE1(arg1, arg2)  CONCATENATE2(arg1, arg2)
#define CONCATENATE2(arg1, arg2)  arg1##arg2

// Expand to number of variadic arguments for up to 8 args.
// VA_NARG(a,b,c)
// PP_ARG_N(a,b,c,8,7,6,5,4,3,2,1,0)
// 3
#define VA_NARG(...) PP_ARG_N(__VA_ARGS__, 8, 7, 6, 5, 4, 3, 2, 1, 0)
#define VA_NARG(...) PP_ARG_N(__VA_ARGS__, 8, 7, 6, 5, 4, 3, 2, 1, 0)
#define PP_ARG_N(_1, _2, _3, _4, _5, _6, _7, _8, N, ...) N

// Arithmetic
#define PLUS_1_1 2
#define PLUS_1_2 3
#define PLUS_1_3 4
#define PLUS_1_4 5
#define PLUS_1_5 6
#define PLUS_1_6 7
#define PLUS_1_7 8
#define PLUS_1(N) CONCATENATE(PLUS_1_, N)

// Field name in struct: __f1 for N = 1
#define AML_FIELD(N) CONCATENATE(__f, N)

// struct fields declaration.
// one field: f1 __f1;
// two fields: f2 __f1; f1 __f2;
// three fields: f3 __f1; f2 __f2; f1 __f3;
// We want fx fields to appear in the order of types provided by users.
// We want __fx names to appear in the reverse order, such that if the user
// wants the second fields it can name it with __f2.
#define AML_DECL_1(N, f1, ...) f1 AML_FIELD(N);
#define AML_DECL_2(N, f2, ...)					\
	f2 AML_FIELD(N); AML_DECL_1(PLUS_1(N), __VA_ARGS__)
#define AML_DECL_3(N, f3, ...)					\
	f3 AML_FIELD(N); AML_DECL_2(PLUS_1(N), __VA_ARGS__)
#define AML_DECL_4(N, f4, ...)					\
	f4 AML_FIELD(N); AML_DECL_3(PLUS_1(N), __VA_ARGS__)
#define AML_DECL_5(N, f5, ...)					\
	f5 AML_FIELD(N); AML_DECL_4(PLUS_1(N), __VA_ARGS__)
#define AML_DECL_6(N, f6, ...)					\
	f6 AML_FIELD(N); AML_DECL_5(PLUS_1(N), __VA_ARGS__)
#define AML_DECL_7(N, f7, ...)					\
	f7 AML_FIELD(N); AML_DECL_6(PLUS_1(N), __VA_ARGS__)
#define AML_DECL_8(N, f8, ...)					\
	f8 AML_FIELD(N); AML_DECL_7(PLUS_1(N), __VA_ARGS__)

// Declare a structure with up to 8 fields.
// (Pick the adequate AML_DECL_ macro and call it.)
#define AML_STRUCT_DECL(...)						\
	struct {							\
	CONCATENATE(AML_DECL_, VA_NARG(__VA_ARGS__))(1, __VA_ARGS__, 0) \
	}

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

#ifdef __cplusplus
}
#endif

#endif //AML_INNER_MALLOC_H
