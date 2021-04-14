/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *******************************************************************************/

#include "aml.h"

// Mapper includes
#include "aml/higher/mapper.h"

// Mapper args (linux)
#include "aml/area/linux.h"
#include "aml/dma/linux-seq.h"

// Mapper args (cuda)
#if AML_HAVE_BACKEND_CUDA
#include "aml/area/cuda.h"
#include "aml/dma/cuda.h"
#endif

//- Struct A Declaration ------------------------------------------------------

struct A {
	size_t val;
};
aml_final_mapper_decl(struct_A_mapper, AML_MAPPER_FLAG_COPY, struct A);

//- Struct B Declaratiion -----------------------------------------------------

struct B {
	int dummy_int;
	double dummy_double;
	struct A *a;
};
aml_mapper_decl(struct_B_mapper,
                AML_MAPPER_FLAG_COPY | AML_MAPPER_FLAG_SPLIT,
                struct B,
                a,
                &struct_A_mapper);

//- Struct C Declaration ------------------------------------------------------

struct C {
	size_t n;
	struct B *b;
};
aml_mapper_decl(struct_C_mapper,
                AML_MAPPER_FLAG_COPY,
                struct C,
                b,
                n,
                &struct_B_mapper);

//- Struct C Declaration ------------------------------------------------------

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

aml_final_mapper_decl(ulong_mapper, AML_MAPPER_FLAG_COPY, unsigned long);

// Largest working mapper decl
aml_mapper_decl(BigStruct_mapper,
                AML_MAPPER_FLAG_COPY,
                struct BigStruct,
                a0,
                na0,
                &ulong_mapper,
                a1,
                na1,
                &ulong_mapper,
                a2,
                na2,
                &ulong_mapper,
                a3,
                na3,
                &ulong_mapper,
                a4,
                na4,
                &ulong_mapper,
                a5,
                na5,
                &ulong_mapper,
                a6,
                na6,
                &ulong_mapper,
                a7,
                na7,
                &ulong_mapper,
                a8,
                na8,
                &ulong_mapper,
                a9,
                na9,
                &ulong_mapper,
                a10,
                na10,
                &ulong_mapper);

//- Equality test + Copy/Free -------------------------------------------------

int eq_struct(struct C *a, struct C *b)
{
	if (a->n != b->n)
		return 0;

	for (size_t i = 0; i < a->n; i++) {
		if (a->b[i].dummy_double != b->b[i].dummy_double)
			return 0;
		if (a->b[i].dummy_int != b->b[i].dummy_int)
			return 0;
		if (a->b[i].a->val != b->b[i].a->val)
			return 0;
	}
	return 1;
}

void test_mapper(struct C *c)
{
	// Linux check
	struct C *host_c;
	assert(aml_mapper_mmap(&struct_C_mapper, c, &host_c, 1, &aml_area_linux,
	                       NULL, aml_dma_linux_sequential, NULL,
	                       NULL) == AML_SUCCESS);
	assert(eq_struct(c, host_c));

	// Cuda check
#if AML_HAVE_BACKEND_CUDA
	if (aml_support_backends(AML_BACKEND_CUDA)) {
		struct C *device_c;
		/* Copy c to cuda device */
		assert(aml_mapper_mmap(&struct_C_mapper, c, &device_c, 1,
		                       &aml_area_cuda, NULL,
		                       &aml_dma_cuda_host_to_device,
		                       aml_dma_cuda_copy_1D,
		                       NULL) == AML_SUCCESS);

		// Change _c to be different from c.
		c->b[0].a->val = 4565467567;

		/* Copy back __c into modified _c */
		assert(aml_mapper_copy(&struct_C_mapper, device_c, c, 1,
		                       &aml_dma_cuda_device_to_host,
		                       aml_dma_cuda_copy_1D,
		                       NULL) == AML_SUCCESS);
		assert(eq_struct(c, host_c));

		aml_mapper_munmap(&struct_C_mapper, device_c, &aml_area_cuda,
		                  &aml_dma_cuda_device_to_host,
		                  aml_dma_cuda_copy_1D, NULL);
	}
#endif
	aml_mapper_munmap(&struct_C_mapper, host_c, &aml_area_linux,
	                  aml_dma_linux_sequential, NULL, NULL);
}

//- Application Data Initialization -------------------------------------------

void init_struct(struct C **_c)
{
	size_t n = 8;

	*_c = AML_INNER_MALLOC_EXTRA(n, struct B, n * sizeof(struct A),
	                             struct C);
	(*_c)->n = n;
	(*_c)->b = AML_INNER_MALLOC_GET_ARRAY((*_c), struct B, struct C);

	for (size_t i = 0; i < n; i++) {
		struct A *a =
		        (struct A *)((size_t)AML_INNER_MALLOC_GET_EXTRA(
		                             (*_c), n, struct B, struct C) +
		                     i * sizeof(struct A));
		a->val = i;
		(*_c)->b[i].a = a;
		(*_c)->b[i].dummy_double = i;
		(*_c)->b[i].dummy_int = -i;
	}
}

int main(int argc, char **argv)
{
	struct C *c;

	// Init
	aml_init(&argc, &argv);
	init_struct(&c);

	// Test
	test_mapper(c);

	// Cleanup
	free(c);
	aml_finalize();
	return 0;
}