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
#include "aml/higher/mapper/deepcopy.h"
#include "aml/higher/mapper/visitor.h"

// Mapper args (linux)
#include "aml/area/linux.h"
#include "aml/dma/linux.h"

// Mapper args (cuda)
#if AML_HAVE_BACKEND_CUDA
#include "aml/area/cuda.h"
#include "aml/dma/cuda.h"
#endif

// Mapper args (cuda)
#if AML_HAVE_BACKEND_HIP
#include "aml/area/hip.h"
#include "aml/dma/hip.h"
#endif

// Structures and mappers declaration. ----------------------------------------

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

// Initialization and teardown. -----------------------------------------------

struct C *c;

void init()
{
	const size_t n = 8;
	c = AML_INNER_MALLOC_EXTRA(n, struct B, n * sizeof(struct A), struct C);
	c->n = n;
	c->b = AML_INNER_MALLOC_GET_ARRAY(c, struct B, struct C);

	for (size_t i = 0; i < n; i++) {
		struct A *a = (struct A *)((size_t)AML_INNER_MALLOC_GET_EXTRA(
		                                   c, n, struct B, struct C) +
		                           i * sizeof(struct A));
		a->val = i;
		c->b[i].a = a;
		c->b[i].dummy_double = i;
		c->b[i].dummy_int = -i;
	}
}

void teardown()
{
	free(c);
}

// Tests ----------------------------------------------------------------------

void test_mapper(struct aml_mapper *mapper,
                 struct aml_area *area,
                 struct aml_area_mmap_options *area_opts,
                 struct aml_dma *dma_dst_host,
                 struct aml_dma *dma_host_dst,
                 aml_dma_operator memcpy_dst_host,
                 aml_dma_operator memcpy_host_dst)
{
	aml_deepcopy_data copy;
	struct aml_mapper_visitor *lhs_vis, *rhs_vis;

	// Make copy.
	assert(aml_mapper_deepcopy(&copy, (void *)c, mapper, area, area_opts,
	                           NULL, dma_host_dst, NULL,
	                           memcpy_host_dst) == AML_SUCCESS);

	// Compare.
	assert(aml_mapper_visitor_create(&lhs_vis, (void *)c, mapper, NULL,
	                                 NULL) == AML_SUCCESS);
	assert(aml_mapper_visitor_create(&rhs_vis, aml_deepcopy_ptr(copy),
	                                 mapper, dma_dst_host,
	                                 memcpy_dst_host) == AML_SUCCESS);
	assert(aml_mapper_visitor_match(lhs_vis, rhs_vis) == 1);

	// Cleanup
	aml_mapper_visitor_destroy(lhs_vis);
	aml_mapper_visitor_destroy(rhs_vis);
	assert(aml_mapper_deepfree(copy) == AML_SUCCESS);
}

//- Main ----------------------------------------------------------------------

int main(int argc, char **argv)
{
	// Init
	aml_init(&argc, &argv);
	init();

	// Tests
	test_mapper(&struct_C_mapper, &aml_area_linux, NULL, aml_dma_linux,
	            aml_dma_linux, aml_dma_linux_memcpy_op,
	            aml_dma_linux_memcpy_op);

#if AML_HAVE_BACKEND_CUDA
	if (aml_support_backends(AML_BACKEND_CUDA)) {
		test_mapper(&struct_C_mapper, &aml_area_cuda, NULL,
		            &aml_dma_cuda, &aml_dma_cuda,
		            aml_dma_cuda_memcpy_op, aml_dma_cuda_memcpy_op);
	}
#endif
#if AML_HAVE_BACKEND_HIP
	if (aml_support_backends(AML_BACKEND_HIP)) {
		test_mapper(&struct_C_mapper, &aml_area_hip, NULL, &aml_dma_hip,
		            &aml_dma_hip, aml_dma_hip_memcpy_op,
		            aml_dma_hip_memcpy_op);
	}
#endif
	// Cleanup
	teardown();
	aml_finalize();
	return 0;
}
