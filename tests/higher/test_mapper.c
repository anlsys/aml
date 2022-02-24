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

// Mapper includes
#include "aml/higher/mapper.h"

// Largest working mapper decl
struct BigStruct {
	unsigned long *a0;
	unsigned na0;
	unsigned long *a1;
	unsigned na1;
	unsigned long *a2;
	unsigned na2;
	unsigned long *a3;
	unsigned na3;
	unsigned long *a4;
	unsigned na4;
	unsigned long *a5;
	unsigned na5;
	unsigned long *a6;
	unsigned na6;
	unsigned long *a7;
	unsigned na7;
	unsigned long *a8;
	unsigned na8;
	unsigned long *a9;
	unsigned na9;
	unsigned long *a10;
	unsigned na10;
};

aml_mapper_decl(BigStruct_mapper,
                0,
                struct BigStruct,
                a0,
                na0,
                &aml_ulong_mapper,
                a1,
                na1,
                &aml_ulong_mapper,
                a2,
                na2,
                &aml_ulong_mapper,
                a3,
                na3,
                &aml_ulong_mapper,
                a4,
                na4,
                &aml_ulong_mapper,
                a5,
                na5,
                &aml_ulong_mapper,
                a6,
                na6,
                &aml_ulong_mapper,
                a7,
                na7,
                &aml_ulong_mapper,
                a8,
                na8,
                &aml_ulong_mapper,
                a9,
                na9,
                &aml_ulong_mapper,
                a10,
                na10,
                &aml_ulong_mapper);

// Other mapped struct.
struct A {
	size_t val;
};
aml_final_mapper_decl(struct_A_mapper, 0, struct A);

struct B {
	int dummy_int;
	double dummy_double;
	struct A *a;
};
aml_mapper_decl(
        struct_B_mapper, AML_MAPPER_FLAG_SPLIT, struct B, a, &struct_A_mapper);

struct C {
	size_t n;
	struct B *b;
};
aml_mapper_decl(struct_C_mapper, 0, struct C, b, n, &struct_B_mapper);

int main(void)
{
	return 0;
}
