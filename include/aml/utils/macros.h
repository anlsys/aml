/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *******************************************************************************/

#ifndef AML_UTILS_MACROS_H
#define AML_UTILS_MACROS_H

/**
 * @defgroup aml_utils_macros "AML Internal Macros helpers"
 * @brief AML helper functions to build macros
 * @{
 *
 **/

// Stringify macro
#define STRINGIFY(a) STRINGIFY_(a)
#define STRINGIFY_(a) #a

// Concatenate two arguments into a macro name
#define CONCATENATE(arg1, arg2) CONCATENATE1(arg1, arg2)
#define CONCATENATE1(arg1, arg2) CONCATENATE2(arg1, arg2)
#define CONCATENATE2(arg1, arg2) arg1##arg2

// Expand to number of variadic arguments for up to 36 args.
// The last argument `_` is here to avoid having empty __VA_ARGS__
// in PP_ARG_N().
#define VA_NARG(...)                                                           \
	PP_ARG_N(__VA_ARGS__, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25,  \
	         24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10,   \
	         9, 8, 7, 6, 5, 4, 3, 2, 1, _)
#define PP_ARG_N(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14,  \
                 _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26,   \
                 _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, N, ...)     \
	N

// Arithmetic
#define PLUS_1_0 1
#define PLUS_1_1 2
#define PLUS_1_2 3
#define PLUS_1_3 4
#define PLUS_1_4 5
#define PLUS_1_5 6
#define PLUS_1_6 7
#define PLUS_1_7 8
#define PLUS_1_8 9
#define PLUS_1_9 10
#define PLUS_1_10 11
#define PLUS_1_11 12
#define PLUS_1_12 13
#define PLUS_1_13 14
#define PLUS_1_14 15
#define PLUS_1_15 16
#define PLUS_1_16 17
#define PLUS_1_17 18
#define PLUS_1_18 19
#define PLUS_1_19 20
#define PLUS_1_20 21
#define PLUS_1_21 22
#define PLUS_1_22 23
#define PLUS_1_23 24
#define PLUS_1_24 25
#define PLUS_1_25 26
#define PLUS_1_26 27
#define PLUS_1_27 28
#define PLUS_1_28 29
#define PLUS_1_29 30
#define PLUS_1_30 31
#define PLUS_1_31 32
#define PLUS_1_32 33
#define PLUS_1_33 34
#define PLUS_1_34 35
#define PLUS_1_35 36
#define PLUS_1_36 37
#define PLUS_1(N) CONCATENATE(PLUS_1_, N)

#define OFFSETOF_1(type, field, ...) offsetof(type, field)
#define OFFSETOF_2(type, field, ...)                                           \
	offsetof(type, field), OFFSETOF_1(type, __VA_ARGS__)
#define OFFSETOF_3(type, field, ...)                                           \
	offsetof(type, field), OFFSETOF_2(type, __VA_ARGS__)
#define OFFSETOF_4(type, field, ...)                                           \
	offsetof(type, field), OFFSETOF_3(type, __VA_ARGS__)
#define OFFSETOF_5(type, field, ...)                                           \
	offsetof(type, field), OFFSETOF_4(type, __VA_ARGS__)
#define OFFSETOF_6(type, field, ...)                                           \
	offsetof(type, field), OFFSETOF_5(type, __VA_ARGS__)
#define OFFSETOF_7(type, field, ...)                                           \
	offsetof(type, field), OFFSETOF_6(type, __VA_ARGS__)
#define OFFSETOF_8(type, field, ...)                                           \
	offsetof(type, field), OFFSETOF_7(type, __VA_ARGS__)
#define OFFSETOF_9(type, field, ...)                                           \
	offsetof(type, field), OFFSETOF_8(type, __VA_ARGS__)
#define OFFSETOF_10(type, field, ...)                                          \
	offsetof(type, field), OFFSETOF_9(type, __VA_ARGS__)
#define OFFSETOF_11(type, field, ...)                                          \
	offsetof(type, field), OFFSETOF_10(type, __VA_ARGS__)
#define OFFSETOF_12(type, field, ...)                                          \
	offsetof(type, field), OFFSETOF_11(type, __VA_ARGS__)
#define OFFSETOF_13(type, field, ...)                                          \
	offsetof(type, field), OFFSETOF_12(type, __VA_ARGS__)
#define OFFSETOF_14(type, field, ...)                                          \
	offsetof(type, field), OFFSETOF_13(type, __VA_ARGS__)
#define OFFSETOF_15(type, field, ...)                                          \
	offsetof(type, field), OFFSETOF_14(type, __VA_ARGS__)
#define OFFSETOF_16(type, field, ...)                                          \
	offsetof(type, field), OFFSETOF_15(type, __VA_ARGS__)
#define OFFSETOF_17(type, field, ...)                                          \
	offsetof(type, field), OFFSETOF_16(type, __VA_ARGS__)
#define OFFSETOF_18(type, field, ...)                                          \
	offsetof(type, field), OFFSETOF_17(type, __VA_ARGS__)
#define OFFSETOF_19(type, field, ...)                                          \
	offsetof(type, field), OFFSETOF_18(type, __VA_ARGS__)
#define OFFSETOF_20(type, field, ...)                                          \
	offsetof(type, field), OFFSETOF_19(type, __VA_ARGS__)
#define OFFSETOF_21(type, field, ...)                                          \
	offsetof(type, field), OFFSETOF_20(type, __VA_ARGS__)
#define OFFSETOF_22(type, field, ...)                                          \
	offsetof(type, field), OFFSETOF_21(type, __VA_ARGS__)
#define OFFSETOF_23(type, field, ...)                                          \
	offsetof(type, field), OFFSETOF_22(type, __VA_ARGS__)
#define OFFSETOF_24(type, field, ...)                                          \
	offsetof(type, field), OFFSETOF_23(type, __VA_ARGS__)
#define OFFSETOF_25(type, field, ...)                                          \
	offsetof(type, field), OFFSETOF_24(type, __VA_ARGS__)
#define OFFSETOF_26(type, field, ...)                                          \
	offsetof(type, field), OFFSETOF_25(type, __VA_ARGS__)
#define OFFSETOF_27(type, field, ...)                                          \
	offsetof(type, field), OFFSETOF_26(type, __VA_ARGS__)
#define OFFSETOF_28(type, field, ...)                                          \
	offsetof(type, field), OFFSETOF_27(type, __VA_ARGS__)
#define OFFSETOF_29(type, field, ...)                                          \
	offsetof(type, field), OFFSETOF_28(type, __VA_ARGS__)
#define OFFSETOF_30(type, field, ...)                                          \
	offsetof(type, field), OFFSETOF_29(type, __VA_ARGS__)
#define OFFSETOF_31(type, field, ...)                                          \
	offsetof(type, field), OFFSETOF_30(type, __VA_ARGS__)
#define OFFSETOF_32(type, field, ...)                                          \
	offsetof(type, field), OFFSETOF_31(type, __VA_ARGS__)
#define OFFSETOF_33(type, field, ...)                                          \
	offsetof(type, field), OFFSETOF_32(type, __VA_ARGS__)
#define OFFSETOF_34(type, field, ...)                                          \
	offsetof(type, field), OFFSETOF_33(type, __VA_ARGS__)
#define OFFSETOF_35(type, field, ...)                                          \
	offsetof(type, field), OFFSETOF_34(type, __VA_ARGS__)
#define OFFSETOF_36(type, field, ...)                                          \
	offsetof(type, field), OFFSETOF_35(type, __VA_ARGS__)
// get offset of fields names in type struct.
#define OFFSETOF(type, ...)                                                    \
	CONCATENATE(OFFSETOF_, VA_NARG(__VA_ARGS__))(type, __VA_ARGS__, _)

#define REP_1(x) x
#define REP_2(x) x, REP_1(x)
#define REP_3(x) x, REP_2(x)
#define REP_4(x) x, REP_3(x)
#define REP_5(x) x, REP_4(x)
#define REP_6(x) x, REP_5(x)
#define REP_7(x) x, REP_6(x)
#define REP_8(x) x, REP_7(x)
#define REP_9(x) x, REP_8(x)
#define REP_10(x) x, REP_9(x)
#define REP_11(x) x, REP_10(x)
#define REP_12(x) x, REP_11(x)
#define REP_13(x) x, REP_12(x)
#define REP_14(x) x, REP_13(x)
#define REP_15(x) x, REP_14(x)
#define REP_16(x) x, REP_15(x)
#define REP_17(x) x, REP_16(x)
#define REP_18(x) x, REP_17(x)
#define REP_19(x) x, REP_18(x)
#define REP_20(x) x, REP_19(x)
#define REP_21(x) x, REP_20(x)
#define REP_22(x) x, REP_21(x)
#define REP_23(x) x, REP_22(x)
#define REP_24(x) x, REP_23(x)
#define REP_25(x) x, REP_24(x)
#define REP_26(x) x, REP_25(x)
#define REP_27(x) x, REP_26(x)
#define REP_28(x) x, REP_27(x)
#define REP_29(x) x, REP_28(x)
#define REP_30(x) x, REP_29(x)
#define REP_31(x) x, REP_30(x)
#define REP_32(x) x, REP_31(x)
#define REP_33(x) x, REP_32(x)
#define REP_34(x) x, REP_33(x)
#define REP_35(x) x, REP_34(x)
#define REP_36(x) x, REP_35(x)
// Replicate x N times.
#define REPLICATE(N, x) CONCATENATE(REP_, N)(x)

//---------------------------------------------------------------------------//
// Malloc utils
//---------------------------------------------------------------------------//

// Field name in struct: __f1 for N = 1
#define AML_FIELD(N) CONCATENATE(__f, N)
#define AML_FIELD_DECL(type, N) type AML_FIELD(N);

// struct fields declaration.
// one field: f1 __f1;
// two fields: f2 __f1; f1 __f2;
// three fields: f3 __f1; f2 __f2; f1 __f3;
// We want fx fields to appear in the order of types provided by users.
// We want __fx names to appear in the reverse order, such that if the user
// wants the second fields it can name it with __f2.
#define AML_DECL_1(N, t, ...) AML_FIELD_DECL(t, N)
#define AML_DECL_2(N, t, ...)                                                  \
	AML_FIELD_DECL(t, N) AML_DECL_1(PLUS_1(N), __VA_ARGS__)
#define AML_DECL_3(N, t, ...)                                                  \
	AML_FIELD_DECL(t, N) AML_DECL_2(PLUS_1(N), __VA_ARGS__)
#define AML_DECL_4(N, t, ...)                                                  \
	AML_FIELD_DECL(t, N) AML_DECL_3(PLUS_1(N), __VA_ARGS__)
#define AML_DECL_5(N, t, ...)                                                  \
	AML_FIELD_DECL(t, N) AML_DECL_4(PLUS_1(N), __VA_ARGS__)
#define AML_DECL_6(N, t, ...)                                                  \
	AML_FIELD_DECL(t, N) AML_DECL_5(PLUS_1(N), __VA_ARGS__)
#define AML_DECL_7(N, t, ...)                                                  \
	AML_FIELD_DECL(t, N) AML_DECL_6(PLUS_1(N), __VA_ARGS__)
#define AML_DECL_8(N, t, ...)                                                  \
	AML_FIELD_DECL(t, N) AML_DECL_7(PLUS_1(N), __VA_ARGS__)
#define AML_DECL_9(N, t, ...)                                                  \
	AML_FIELD_DECL(t, N) AML_DECL_8(PLUS_1(N), __VA_ARGS__)
#define AML_DECL_10(N, t, ...)                                                 \
	AML_FIELD_DECL(t, N) AML_DECL_9(PLUS_1(N), __VA_ARGS__)
#define AML_DECL_11(N, t, ...)                                                 \
	AML_FIELD_DECL(t, N) AML_DECL_10(PLUS_1(N), __VA_ARGS__)
#define AML_DECL_12(N, t, ...)                                                 \
	AML_FIELD_DECL(t, N) AML_DECL_11(PLUS_1(N), __VA_ARGS__)
#define AML_DECL_13(N, t, ...)                                                 \
	AML_FIELD_DECL(t, N) AML_DECL_12(PLUS_1(N), __VA_ARGS__)
#define AML_DECL_14(N, t, ...)                                                 \
	AML_FIELD_DECL(t, N) AML_DECL_13(PLUS_1(N), __VA_ARGS__)
#define AML_DECL_15(N, t, ...)                                                 \
	AML_FIELD_DECL(t, N) AML_DECL_14(PLUS_1(N), __VA_ARGS__)
#define AML_DECL_16(N, t, ...)                                                 \
	AML_FIELD_DECL(t, N) AML_DECL_15(PLUS_1(N), __VA_ARGS__)
#define AML_DECL_17(N, t, ...)                                                 \
	AML_FIELD_DECL(t, N) AML_DECL_16(PLUS_1(N), __VA_ARGS__)
#define AML_DECL_18(N, t, ...)                                                 \
	AML_FIELD_DECL(t, N) AML_DECL_17(PLUS_1(N), __VA_ARGS__)
#define AML_DECL_19(N, t, ...)                                                 \
	AML_FIELD_DECL(t, N) AML_DECL_18(PLUS_1(N), __VA_ARGS__)
#define AML_DECL_20(N, t, ...)                                                 \
	AML_FIELD_DECL(t, N) AML_DECL_19(PLUS_1(N), __VA_ARGS__)
#define AML_DECL_21(N, t, ...)                                                 \
	AML_FIELD_DECL(t, N) AML_DECL_20(PLUS_1(N), __VA_ARGS__)
#define AML_DECL_22(N, t, ...)                                                 \
	AML_FIELD_DECL(t, N) AML_DECL_21(PLUS_1(N), __VA_ARGS__)
#define AML_DECL_23(N, t, ...)                                                 \
	AML_FIELD_DECL(t, N) AML_DECL_22(PLUS_1(N), __VA_ARGS__)
#define AML_DECL_24(N, t, ...)                                                 \
	AML_FIELD_DECL(t, N) AML_DECL_23(PLUS_1(N), __VA_ARGS__)
#define AML_DECL_25(N, t, ...)                                                 \
	AML_FIELD_DECL(t, N) AML_DECL_24(PLUS_1(N), __VA_ARGS__)
#define AML_DECL_26(N, t, ...)                                                 \
	AML_FIELD_DECL(t, N) AML_DECL_25(PLUS_1(N), __VA_ARGS__)
#define AML_DECL_27(N, t, ...)                                                 \
	AML_FIELD_DECL(t, N) AML_DECL_26(PLUS_1(N), __VA_ARGS__)
#define AML_DECL_28(N, t, ...)                                                 \
	AML_FIELD_DECL(t, N) AML_DECL_27(PLUS_1(N), __VA_ARGS__)
#define AML_DECL_29(N, t, ...)                                                 \
	AML_FIELD_DECL(t, N) AML_DECL_28(PLUS_1(N), __VA_ARGS__)
#define AML_DECL_30(N, t, ...)                                                 \
	AML_FIELD_DECL(t, N) AML_DECL_29(PLUS_1(N), __VA_ARGS__)
#define AML_DECL_31(N, t, ...)                                                 \
	AML_FIELD_DECL(t, N) AML_DECL_30(PLUS_1(N), __VA_ARGS__)
#define AML_DECL_32(N, t, ...)                                                 \
	AML_FIELD_DECL(t, N) AML_DECL_31(PLUS_1(N), __VA_ARGS__)
#define AML_DECL_33(N, t, ...)                                                 \
	AML_FIELD_DECL(t, N) AML_DECL_32(PLUS_1(N), __VA_ARGS__)
#define AML_DECL_34(N, t, ...)                                                 \
	AML_FIELD_DECL(t, N) AML_DECL_33(PLUS_1(N), __VA_ARGS__)
#define AML_DECL_35(N, t, ...)                                                 \
	AML_FIELD_DECL(t, N) AML_DECL_34(PLUS_1(N), __VA_ARGS__)
#define AML_DECL_36(N, t, ...)                                                 \
	AML_FIELD_DECL(t, N) AML_DECL_35(PLUS_1(N), __VA_ARGS__)

// Declare a structure with up to 36 fields.
// (Pick the adequate AML_DECL_ macro and call it.)
#define AML_STRUCT_DECL(...)                                                   \
	struct {                                                               \
		CONCATENATE(AML_DECL_, VA_NARG(__VA_ARGS__))                   \
		(1, __VA_ARGS__, 0)                                            \
	}

/** Returns the size required for allocation of up to 8 types **/
#define AML_SIZEOF_ALIGNED(...) sizeof(AML_STRUCT_DECL(__VA_ARGS__))

/**
 * Returns the size required for allocation of up to 7 types plus one array.
 * @param n: The number of elements in array.
 * @param type: The type of array elements.
 * @param ...: Up to 7 fields type preceding array allocation space.
 **/
#define AML_SIZEOF_ALIGNED_ARRAY(n, type, ...)                                 \
	(sizeof(AML_STRUCT_DECL(__VA_ARGS__, type)) + ((n)-1) * sizeof(type))

/** Returns the offset of the nth type of a list of up to 8 types. **/
#define AML_OFFSETOF_ALIGNED(N, ...)                                           \
	offsetof(AML_STRUCT_DECL(__VA_ARGS__), AML_FIELD(N))

//---------------------------------------------------------------------------//
// Mapper utils
//---------------------------------------------------------------------------//

// Returns a number that can be combine witrh CONCATENATE to select
// The good macro to call. If the number of argument is a multiple of 3
// then it returns 3, a multiple of 2, then it returns 2, 0 arguments,
// then returns 0, else 1.
#define __AML_MAPPER_DECL_SELECT(...)                                          \
	PP_ARG_N(__VA_ARGS__, 3, 1, 2, 3, 2, 1, 3, 1, 2, 3, 2, 1, 3, 1, 2, 3,  \
	         2, 11, 3, 1, 2, 3, 2, 1, 3, 1, 2, 3, 2, 1, 3, 1, 2, 3, 2, 1,  \
	         _)

// Fills a static mapper structure at declaration.
#define __AML_MAPPER_INIT(type, nf, off, num, fds)                             \
	{                                                                      \
		.size = sizeof(type), .n_fields = nf, .offsets = off,          \
		.num_elements = num, .fields = fds                             \
	}

#define __AML_MAPPER_DECL_1(...)                                                 \
	"Invalid number of arguments for macro aml_mapper_decl(). "              \
	"Must be (name, type, ...) where ... is either no field argument, or a " \
	"multiple of 2 or a multiple of 3 arguments"

#define EVEN_2(a, b) a
#define EVEN_4(a, b, ...) a, EVEN_2(__VA_ARGS__)
#define EVEN_6(a, b, ...) a, EVEN_4(__VA_ARGS__)
#define EVEN_8(a, b, ...) a, EVEN_6(__VA_ARGS__)
#define EVEN_10(a, b, ...) a, EVEN_8__VA_ARGS__)
#define EVEN_12(a, b, ...) a, EVEN_10(__VA_ARGS__)
#define EVEN_14(a, b, ...) a, EVEN_12(__VA_ARGS__)
#define EVEN_16(a, b, ...) a, EVEN_14(__VA_ARGS__)
#define EVEN_18(a, b, ...) a, EVEN_16(__VA_ARGS__)
#define EVEN_20(a, b, ...) a, EVEN_18(__VA_ARGS__)
#define EVEN_22(a, b, ...) a, EVEN_20(__VA_ARGS__)
#define EVEN_24(a, b, ...) a, EVEN_22(__VA_ARGS__)
#define EVEN_26(a, b, ...) a, EVEN_24(__VA_ARGS__)
#define EVEN_28(a, b, ...) a, EVEN_26(__VA_ARGS__)
#define EVEN_30(a, b, ...) a, EVEN_28(__VA_ARGS__)
#define EVEN_32(a, b, ...) a, EVEN_30(__VA_ARGS__)
#define EVEN_34(a, b, ...) a, EVEN_32(__VA_ARGS__)
#define EVEN_36(a, b, ...) a, EVEN_34(__VA_ARGS__)
// Even elements
#define EVEN(...) CONCATENATE(EVEN_, VA_NARG(__VA_ARGS__))(__VA_ARGS__)

#define ODD_2(a, b) b
#define ODD_4(a, b, ...) b, ODD_2(__VA_ARGS__)
#define ODD_6(a, b, ...) b, ODD_4(__VA_ARGS__)
#define ODD_8(a, b, ...) b, ODD_6(__VA_ARGS__)
#define ODD_10(a, b, ...) b, ODD_8(__VA_ARGS__)
#define ODD_12(a, b, ...) b, ODD_10(__VA_ARGS__)
#define ODD_14(a, b, ...) b, ODD_12(__VA_ARGS__)
#define ODD_16(a, b, ...) b, ODD_14(__VA_ARGS__)
#define ODD_18(a, b, ...) b, ODD_16(__VA_ARGS__)
#define ODD_20(a, b, ...) b, ODD_18(__VA_ARGS__)
#define ODD_22(a, b, ...) b, ODD_20(__VA_ARGS__)
#define ODD_24(a, b, ...) b, ODD_22(__VA_ARGS__)
#define ODD_26(a, b, ...) b, ODD_24(__VA_ARGS__)
#define ODD_28(a, b, ...) b, ODD_26(__VA_ARGS__)
#define ODD_30(a, b, ...) b, ODD_28(__VA_ARGS__)
#define ODD_32(a, b, ...) b, ODD_30(__VA_ARGS__)
#define ODD_34(a, b, ...) b, ODD_32(__VA_ARGS__)
#define ODD_36(a, b, ...) b, ODD_34(__VA_ARGS__)
// Odd elements
#define ODD(...) CONCATENATE(ODD_, VA_NARG(__VA_ARGS__))(__VA_ARGS__)

// Number of pairs
#define PAIR_N(...) VA_NARG(ODD(__VA_ARGS__))

/**
 * Declare a mapper with pointers to descend.
 * @arg name: The name of the variable mapper to create.
 * @arg type: The type of the element to map.
 * @arg field: The name of the first field to map in `type` struct.
 * @arg field_mapper: A pointer to a `struct aml_mapper` that maps
 * the field type.
 * @arg __VA_ARGS__: A list of (field, field_mapper).
 */
#define __AML_MAPPER_DECL_2(name, type, ...)                                   \
	num_element_fn __NUM_ELEMENTS_FN_##name[PAIR_N(__VA_ARGS__)] = {       \
	        REPLICATE(PAIR_N(__VA_ARGS__), NULL)};                         \
	size_t __OFFSETS_##name[PAIR_N(__VA_ARGS__)] = {                       \
	        OFFSETOF(type, EVEN(__VA_ARGS__))};                            \
	struct aml_mapper *__FIELDS_##name[PAIR_N(__VA_ARGS__)] = {            \
	        ODD(__VA_ARGS__)};                                             \
	struct aml_mapper name =                                               \
	        __AML_MAPPER_INIT(type, PAIR_N(__VA_ARGS__), __OFFSETS_##name, \
	                          __NUM_ELEMENTS_FN_##name, __FIELDS_##name)

#define TRIPLE_1_3(a, b, c) a
#define TRIPLE_1_6(a, b, c, ...) a, TRIPLE_1_3(__VA_ARGS__)
#define TRIPLE_1_9(a, b, c, ...) a, TRIPLE_1_6(__VA_ARGS__)
#define TRIPLE_1_12(a, b, c, ...) a, TRIPLE_1_9(__VA_ARGS__)
#define TRIPLE_1_15(a, b, c, ...) a, TRIPLE_1_12(__VA_ARGS__)
#define TRIPLE_1_18(a, b, c, ...) a, TRIPLE_1_15(__VA_ARGS__)
#define TRIPLE_1_21(a, b, c, ...) a, TRIPLE_1_18(__VA_ARGS__)
#define TRIPLE_1_24(a, b, c, ...) a, TRIPLE_1_21(__VA_ARGS__)
#define TRIPLE_1_27(a, b, c, ...) a, TRIPLE_1_24(__VA_ARGS__)
#define TRIPLE_1_30(a, b, c, ...) a, TRIPLE_1_27(__VA_ARGS__)
#define TRIPLE_1_33(a, b, c, ...) a, TRIPLE_1_30(__VA_ARGS__)
#define TRIPLE_1_36(a, b, c, ...) a, TRIPLE_1_33(__VA_ARGS__)

#define TRIPLE_2_3(a, b, c) b
#define TRIPLE_2_6(a, b, c, ...) b, TRIPLE_2_3(__VA_ARGS__)
#define TRIPLE_2_9(a, b, c, ...) b, TRIPLE_2_6(__VA_ARGS__)
#define TRIPLE_2_12(a, b, c, ...) b, TRIPLE_2_9(__VA_ARGS__)
#define TRIPLE_2_15(a, b, c, ...) b, TRIPLE_2_12(__VA_ARGS__)
#define TRIPLE_2_18(a, b, c, ...) b, TRIPLE_2_15(__VA_ARGS__)
#define TRIPLE_2_21(a, b, c, ...) b, TRIPLE_2_18(__VA_ARGS__)
#define TRIPLE_2_24(a, b, c, ...) b, TRIPLE_2_21(__VA_ARGS__)
#define TRIPLE_2_27(a, b, c, ...) b, TRIPLE_2_24(__VA_ARGS__)
#define TRIPLE_2_30(a, b, c, ...) b, TRIPLE_2_27(__VA_ARGS__)
#define TRIPLE_2_33(a, b, c, ...) b, TRIPLE_2_30(__VA_ARGS__)
#define TRIPLE_2_36(a, b, c, ...) b, TRIPLE_2_33(__VA_ARGS__)

#define TRIPLE_3_3(a, b, c) c
#define TRIPLE_3_6(a, b, c, ...) c, TRIPLE_3_3(__VA_ARGS__)
#define TRIPLE_3_9(a, b, c, ...) c, TRIPLE_3_6(__VA_ARGS__)
#define TRIPLE_3_12(a, b, c, ...) c, TRIPLE_3_9(__VA_ARGS__)
#define TRIPLE_3_15(a, b, c, ...) c, TRIPLE_3_12(__VA_ARGS__)
#define TRIPLE_3_18(a, b, c, ...) c, TRIPLE_3_15(__VA_ARGS__)
#define TRIPLE_3_21(a, b, c, ...) c, TRIPLE_3_18(__VA_ARGS__)
#define TRIPLE_3_24(a, b, c, ...) c, TRIPLE_3_21(__VA_ARGS__)
#define TRIPLE_3_27(a, b, c, ...) c, TRIPLE_3_24(__VA_ARGS__)
#define TRIPLE_3_30(a, b, c, ...) c, TRIPLE_3_27(__VA_ARGS__)
#define TRIPLE_3_33(a, b, c, ...) c, TRIPLE_3_30(__VA_ARGS__)
#define TRIPLE_3_36(a, b, c, ...) c, TRIPLE_3_33(__VA_ARGS__)

// Select first element of each triplet
#define TRIPLE_1(...) CONCATENATE(TRIPLE_1_, VA_NARG(__VA_ARGS__))(__VA_ARGS__)
// Select second element of each triplet
#define TRIPLE_2(...) CONCATENATE(TRIPLE_2_, VA_NARG(__VA_ARGS__))(__VA_ARGS__)
// Select third element of each triplet
#define TRIPLE_3(...) CONCATENATE(TRIPLE_3_, VA_NARG(__VA_ARGS__))(__VA_ARGS__)
// Number of triplet
#define TRIPLE_N(...) VA_NARG(TRIPLE_3(__VA_ARGS__))

#define __AML_DECLARE_NELEM_FN_1(name, type, field)                            \
	size_t name##_num_element_fn_##field(void *ptr)                        \
	{                                                                      \
		return *(size_t *)((size_t)ptr + offsetof(type, field));       \
	}
#define __AML_DECLARE_NELEM_FN_2(name, type, field, ...)                       \
	__AML_DECLARE_NELEM_FN_1(name, type, field)                            \
	__AML_DECLARE_NELEM_FN_1(name, type, __VA_ARGS__)
#define __AML_DECLARE_NELEM_FN_3(name, type, field, ...)                       \
	__AML_DECLARE_NELEM_FN_1(name, type, field)                            \
	__AML_DECLARE_NELEM_FN_2(name, type, __VA_ARGS__)
#define __AML_DECLARE_NELEM_FN_4(name, type, field, ...)                       \
	__AML_DECLARE_NELEM_FN_1(name, type, field)                            \
	__AML_DECLARE_NELEM_FN_3(name, type, __VA_ARGS__)
#define __AML_DECLARE_NELEM_FN_5(name, type, field, ...)                       \
	__AML_DECLARE_NELEM_FN_1(name, type, field)                            \
	__AML_DECLARE_NELEM_FN_4(name, type, __VA_ARGS__)
#define __AML_DECLARE_NELEM_FN_6(name, type, field, ...)                       \
	__AML_DECLARE_NELEM_FN_1(name, type, field)                            \
	__AML_DECLARE_NELEM_FN_5(name, type, __VA_ARGS__)
#define __AML_DECLARE_NELEM_FN_7(name, type, field, ...)                       \
	__AML_DECLARE_NELEM_FN_1(name, type, field)                            \
	__AML_DECLARE_NELEM_FN_6(name, type, __VA_ARGS__)
#define __AML_DECLARE_NELEM_FN_8(name, type, field, ...)                       \
	__AML_DECLARE_NELEM_FN_1(name, type, field)                            \
	__AML_DECLARE_NELEM_FN_7(name, type, __VA_ARGS__)
#define __AML_DECLARE_NELEM_FN_9(name, type, field, ...)                       \
	__AML_DECLARE_NELEM_FN_1(name, type, field)                            \
	__AML_DECLARE_NELEM_FN_8(name, type, __VA_ARGS__)
#define __AML_DECLARE_NELEM_FN_10(name, type, field, ...)                      \
	__AML_DECLARE_NELEM_FN_1(name, type, field)                            \
	__AML_DECLARE_NELEM_FN_9(name, type, __VA_ARGS__)
#define __AML_DECLARE_NELEM_FN_11(name, type, field, ...)                      \
	__AML_DECLARE_NELEM_FN_1(name, type, field)                            \
	__AML_DECLARE_NELEM_FN_10(name, type, __VA_ARGS__)
#define __AML_DECLARE_NELEM_FN_12(name, type, field, ...)                      \
	__AML_DECLARE_NELEM_FN_1(name, type, field)                            \
	__AML_DECLARE_NELEM_FN_11(name, type, __VA_ARGS__)
#define __AML_DECLARE_NELEM_FN_13(name, type, field, ...)                      \
	__AML_DECLARE_NELEM_FN_1(name, type, field)                            \
	__AML_DECLARE_NELEM_FN_12(name, type, __VA_ARGS__)
#define __AML_DECLARE_NELEM_FN(name, type, ...)                                \
	CONCATENATE(__AML_DECLARE_NELEM_FN_, VA_NARG(__VA_ARGS__))             \
	(name, type, __VA_ARGS__)

#define __AML_NELEM_FN_1(name, field) name##_num_element_fn_##field
#define __AML_NELEM_FN_2(name, field, ...)                                     \
	name##_num_element_fn_##field, __AML_NELEM_FN_1(name, __VA_ARGS__)
#define __AML_NELEM_FN_3(name, field, ...)                                     \
	name##_num_element_fn_##field, __AML_NELEM_FN_2(name, __VA_ARGS__)
#define __AML_NELEM_FN_4(name, field, ...)                                     \
	name##_num_element_fn_##field, __AML_NELEM_FN_3(name, __VA_ARGS__)
#define __AML_NELEM_FN_5(name, field, ...)                                     \
	name##_num_element_fn_##field, __AML_NELEM_FN_4(name, __VA_ARGS__)
#define __AML_NELEM_FN_6(name, field, ...)                                     \
	name##_num_element_fn_##field, __AML_NELEM_FN_5(name, __VA_ARGS__)
#define __AML_NELEM_FN_7(name, field, ...)                                     \
	name##_num_element_fn_##field, __AML_NELEM_FN_6(name, __VA_ARGS__)
#define __AML_NELEM_FN_8(name, field, ...)                                     \
	name##_num_element_fn_##field, __AML_NELEM_FN_7(name, __VA_ARGS__)
#define __AML_NELEM_FN_9(name, field, ...)                                     \
	name##_num_element_fn_##field, __AML_NELEM_FN_8(name, __VA_ARGS__)
#define __AML_NELEM_FN_10(name, field, ...)                                    \
	name##_num_element_fn_##field, __AML_NELEM_FN_9(name, __VA_ARGS__)
#define __AML_NELEM_FN_11(name, field, ...)                                    \
	name##_num_element_fn_##field, __AML_NELEM_FN_10(name, __VA_ARGS__)
#define __AML_NELEM_FN_12(name, field, ...)                                    \
	name##_num_element_fn_##field, __AML_NELEM_FN_11(name, __VA_ARGS__)
#define __AML_NELEM_FN_13(name, field, ...)                                    \
	name##_num_element_fn_##field, __AML_NELEM_FN_12(name, __VA_ARGS__)
#define __AML_NELEM_FN(name, ...)                                              \
	CONCATENATE(__AML_NELEM_FN_, VA_NARG(__VA_ARGS__))                     \
	(name, __VA_ARGS__)

/**
 * Declare a mapper with pointers to descend.
 * @arg name: The name of the variable mapper to create.
 * @arg type: The type of the element to map.
 * @arg field: The name of the first field to map in `type` struct.
 * @arg num_elements: The name of the struct field that counts the number
 * of struct `field` elements pointed by  struct field `field`.
 * @arg field_mapper: A pointer to a `struct aml_mapper` that maps
 * the field type.
 * @arg __VA_ARGS__: A list of (field, num_elements, field_mapper).
 */
#define __AML_MAPPER_DECL_3(name, type, ...)                                   \
	__AML_DECLARE_NELEM_FN(name, type, TRIPLE_2(__VA_ARGS__))              \
	num_element_fn __NUM_ELEMENTS_FN_##name[TRIPLE_N(__VA_ARGS__)] = {     \
	        __AML_NELEM_FN(name, TRIPLE_2(__VA_ARGS__))};                  \
	size_t __OFFSETS_##name[TRIPLE_N(__VA_ARGS__)] = {                     \
	        OFFSETOF(type, TRIPLE_1(__VA_ARGS__))};                        \
	struct aml_mapper *__FIELDS_##name[TRIPLE_N(__VA_ARGS__)] = {          \
	        TRIPLE_3(__VA_ARGS__)};                                        \
	struct aml_mapper name = __AML_MAPPER_INIT(                            \
	        type, TRIPLE_N(__VA_ARGS__), __OFFSETS_##name,                 \
	        __NUM_ELEMENTS_FN_##name, __FIELDS_##name)

/**
 * @}
 **/

#endif // AML_UTILS_MACROS_H
