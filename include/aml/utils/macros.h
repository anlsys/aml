/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#ifndef AML_MACRO_H
#define AML_MACRO_H

// Stringify macro
#define STRINGIFY(a) STRINGIFY_(a)
#define STRINGIFY_(a) #a

// Concatenate two arguments into a macro name
#define CONCATENATE(arg1, arg2) CONCATENATE1(arg1, arg2)
#define CONCATENATE1(arg1, arg2) CONCATENATE2(arg1, arg2)
#define CONCATENATE2(arg1, arg2) arg1##arg2

// Expand to number of variadic arguments for up to 8 args.
// VA_NARG(a,b,c)
// PP_ARG_N(a,b,c,8,7,6,5,4,3,2,1,0)
// 3
#define VA_NARG(...)                                                           \
	PP_ARG_N(__VA_ARGS__, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, \
		 2, 1, 0)
#define PP_ARG_N(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14,  \
		 _15, _16, N, ...)                                             \
	N

// Add one
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
#define PLUS_1(N) CONCATENATE(PLUS_1_, N)

// Subtract one
#define MINUS_1_1 0
#define MINUS_1_2 1
#define MINUS_1_3 2
#define MINUS_1_4 3
#define MINUS_1_5 4
#define MINUS_1_6 5
#define MINUS_1_7 6
#define MINUS_1_8 7
#define MINUS_1_9 8
#define MINUS_1_10 9
#define MINUS_1_11 10
#define MINUS_1_12 11
#define MINUS_1_13 12
#define MINUS_1_14 13
#define MINUS_1_15 14
#define MINUS_1_16 15
#define MINUS_1(N) CONCATENATE(MINUS_1_, N)

// Concatenate up to 16 elements separated with a comma.
#define LIST_1(x, ...) x
#define LIST_2(x, ...) x, LIST_1(__VA_ARGS__)
#define LIST_3(x, ...) x, LIST_2(__VA_ARGS__)
#define LIST_4(x, ...) x, LIST_3(__VA_ARGS__)
#define LIST_5(x, ...) x, LIST_4(__VA_ARGS__)
#define LIST_6(x, ...) x, LIST_5(__VA_ARGS__)
#define LIST_7(x, ...) x, LIST_6(__VA_ARGS__)
#define LIST_8(x, ...) x, LIST_7(__VA_ARGS__)
#define LIST_9(x, ...) x, LIST_8(__VA_ARGS__)
#define LIST_10(x, ...) x, LIST_9(__VA_ARGS__)
#define LIST_11(x, ...) x, LIST_10(__VA_ARGS__)
#define LIST_12(x, ...) x, LIST_11(__VA_ARGS__)
#define LIST_13(x, ...) x, LIST_12(__VA_ARGS__)
#define LIST_14(x, ...) x, LIST_13(__VA_ARGS__)
#define LIST_15(x, ...) x, LIST_14(__VA_ARGS__)
#define LIST_16(x, ...) x, LIST_15(__VA_ARGS__)
#define LIST(N, ...) CONCATENATE(LIST_, N)(__VA_ARGS__)

// Skip up to 16 arguments.
#define SKIP_1(x, ...) __VA_ARGS__
#define SKIP_2(x, ...) SKIP_1(__VA_ARGS__)
#define SKIP_3(x, ...) SKIP_2(__VA_ARGS__)
#define SKIP_4(x, ...) SKIP_3(__VA_ARGS__)
#define SKIP_5(x, ...) SKIP_4(__VA_ARGS__)
#define SKIP_6(x, ...) SKIP_5(__VA_ARGS__)
#define SKIP_7(x, ...) SKIP_6(__VA_ARGS__)
#define SKIP_8(x, ...) SKIP_7(__VA_ARGS__)
#define SKIP_9(x, ...) SKIP_8(__VA_ARGS__)
#define SKIP_10(x, ...) SKIP_9(__VA_ARGS__)
#define SKIP_11(x, ...) SKIP_10(__VA_ARGS__)
#define SKIP_12(x, ...) SKIP_11(__VA_ARGS__)
#define SKIP_13(x, ...) SKIP_12(__VA_ARGS__)
#define SKIP_14(x, ...) SKIP_13(__VA_ARGS__)
#define SKIP_15(x, ...) SKIP_14(__VA_ARGS__)
#define SKIP_16(x, ...) SKIP_15(__VA_ARGS__)
#define SKIP(N, ...) CONCATENATE(SKIP_, N)(__VA_ARGS__)

/** Declare and initialize a static array of up to 16 elements **/
#define STATIC_ARRAY_DECL(type, name, N, ...)                                  \
	type name[N] = {LIST(N, __VA_ARGS__)}

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
#define AML_DECL_2(N, f2, ...)                                                 \
	f2 AML_FIELD(N);                                                       \
	AML_DECL_1(PLUS_1(N), __VA_ARGS__)
#define AML_DECL_3(N, f3, ...)                                                 \
	f3 AML_FIELD(N);                                                       \
	AML_DECL_2(PLUS_1(N), __VA_ARGS__)
#define AML_DECL_4(N, f4, ...)                                                 \
	f4 AML_FIELD(N);                                                       \
	AML_DECL_3(PLUS_1(N), __VA_ARGS__)
#define AML_DECL_5(N, f5, ...)                                                 \
	f5 AML_FIELD(N);                                                       \
	AML_DECL_4(PLUS_1(N), __VA_ARGS__)
#define AML_DECL_6(N, f6, ...)                                                 \
	f6 AML_FIELD(N);                                                       \
	AML_DECL_5(PLUS_1(N), __VA_ARGS__)
#define AML_DECL_7(N, f7, ...)                                                 \
	f7 AML_FIELD(N);                                                       \
	AML_DECL_6(PLUS_1(N), __VA_ARGS__)
#define AML_DECL_8(N, f8, ...)                                                 \
	f8 AML_FIELD(N);                                                       \
	AML_DECL_7(PLUS_1(N), __VA_ARGS__)
#define AML_DECL_9(N, f9, ...)                                                 \
	f9 AML_FIELD(N);                                                       \
	AML_DECL_8(PLUS_1(N), __VA_ARGS__)
#define AML_DECL_10(N, f10, ...)                                               \
	f10 AML_FIELD(N);                                                      \
	AML_DECL_9(PLUS_1(N), __VA_ARGS__)
#define AML_DECL_11(N, f11, ...)                                               \
	f11 AML_FIELD(N);                                                      \
	AML_DECL_10(PLUS_1(N), __VA_ARGS__)
#define AML_DECL_12(N, f12, ...)                                               \
	f12 AML_FIELD(N);                                                      \
	AML_DECL_11(PLUS_1(N), __VA_ARGS__)
#define AML_DECL_13(N, f13, ...)                                               \
	f13 AML_FIELD(N);                                                      \
	AML_DECL_12(PLUS_1(N), __VA_ARGS__)
#define AML_DECL_14(N, f14, ...)                                               \
	f14 AML_FIELD(N);                                                      \
	AML_DECL_13(PLUS_1(N), __VA_ARGS__)
#define AML_DECL_15(N, f15, ...)                                               \
	f15 AML_FIELD(N);                                                      \
	AML_DECL_14(PLUS_1(N), __VA_ARGS__)
#define AML_DECL_16(N, f16, ...)                                               \
	f16 AML_FIELD(N);                                                      \
	AML_DECL_15(PLUS_1(N), __VA_ARGS__)

// Declare a structure with up to 16 fields.
// (Pick the adequate AML_DECL_ macro and call it.)
#define AML_STRUCT_DECL(...)                                                   \
	struct {                                                               \
		CONCATENATE(AML_DECL_, VA_NARG(__VA_ARGS__))                   \
		(1, __VA_ARGS__, 0)                                            \
	}

#endif // AML_MACRO_H
