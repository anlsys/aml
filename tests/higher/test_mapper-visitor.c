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
#include "aml/higher/mapper/visitor.h"

// Mapper args (linux)
#include "aml/dma/linux.h"

struct A {
	size_t unused;
};
aml_final_mapper_decl(struct_A_mapper, 0, struct A);

struct B {
	struct A *first;
	struct A *second;
};
aml_mapper_decl(struct_B_mapper,
                0,
                struct B,
                first,
                &struct_A_mapper,
                second,
                &struct_A_mapper);

struct C {
	size_t n;
	struct B *first;
};

aml_mapper_decl(struct_C_mapper, 0, struct C, first, n, &struct_B_mapper);

struct C *c;

void init()
{
	const size_t n = 2;
	c = AML_INNER_MALLOC_EXTRA(n, struct B, n * sizeof(struct A), struct C);
	c->n = n;
	c->first = AML_INNER_MALLOC_GET_ARRAY(c, struct B, struct C);

	for (size_t i = 0; i < n; i++) {
		struct A *a = (struct A *)((size_t)AML_INNER_MALLOC_GET_EXTRA(
		                                   c, n, struct B, struct C) +
		                           i * sizeof(struct A));
		c->first[i].first = a;
		c->first[i].second = a;
	}
}

void teardown()
{
	free(c);
}

void test_mapper_visitor()
{
	struct aml_mapper_visitor *it;

	assert(aml_mapper_visitor_create(
	               &it, (void *)c, &struct_C_mapper, aml_dma_linux,
	               aml_dma_linux_memcpy_op) == AML_SUCCESS);

	// Root
	assert(aml_mapper_visitor_ptr(it) == c);
	assert(aml_mapper_visitor_next_field(it) == -AML_EDOM);
	assert(aml_mapper_visitor_prev_field(it) == -AML_EDOM);
	assert(aml_mapper_visitor_next_array_element(it) == -AML_EDOM);
	assert(aml_mapper_visitor_prev_array_element(it) == -AML_EDOM);
	assert(!aml_mapper_visitor_is_array(it));
	assert(aml_mapper_visitor_first_field(it) == AML_SUCCESS);

	// field of 2 struct B
	assert(aml_mapper_visitor_ptr(it) == c->first);
	assert(aml_mapper_visitor_array_len(it) == c->n);
	assert(aml_mapper_visitor_next_field(it) == -AML_EDOM);
	assert(aml_mapper_visitor_prev_field(it) == -AML_EDOM);
	assert(aml_mapper_visitor_prev_array_element(it) == -AML_EDOM);
	assert(aml_mapper_visitor_next_array_element(it) == AML_SUCCESS);
	assert(aml_mapper_visitor_next_array_element(it) == -AML_EDOM);
	assert(aml_mapper_visitor_ptr(it) == &(c->first[1]));
	assert(aml_mapper_visitor_parent(it) == AML_SUCCESS);
	assert(aml_mapper_visitor_ptr(it) == c);
	assert(aml_mapper_visitor_first_field(it) == AML_SUCCESS);
	assert(aml_mapper_visitor_first_field(it) == AML_SUCCESS);

	// Struct A: first field of struct B.
	assert(aml_mapper_visitor_ptr(it) == c->first[0].first);
	assert(!aml_mapper_visitor_is_array(it));
	assert(aml_mapper_visitor_first_field(it) == -AML_EDOM);
	assert(aml_mapper_visitor_prev_field(it) == -AML_EDOM);
	assert(aml_mapper_visitor_next_array_element(it) == -AML_EDOM);
	assert(aml_mapper_visitor_prev_array_element(it) == -AML_EDOM);
	assert(aml_mapper_visitor_next_field(it) == AML_SUCCESS);

	// Struct A: second field of struct B.
	assert(aml_mapper_visitor_ptr(it) == c->first[0].second);
	assert(!aml_mapper_visitor_is_array(it));
	assert(aml_mapper_visitor_first_field(it) == -AML_EDOM);
	assert(aml_mapper_visitor_next_field(it) == -AML_EDOM);

	// Cleanup
	aml_mapper_visitor_destroy(it);
}

void test_mapper_visitor_match()
{
	struct aml_mapper_visitor *lhs, *rhs;

	aml_mapper_visitor_create(&lhs, (void *)c, &struct_C_mapper,
	                          aml_dma_linux, aml_dma_linux_memcpy_op);
	aml_mapper_visitor_create(&rhs, (void *)c, &struct_C_mapper,
	                          aml_dma_linux, aml_dma_linux_memcpy_op);

	// Test equality.
	assert(aml_mapper_visitor_match(lhs, rhs) == 1);

	// Cleanup
	aml_mapper_visitor_destroy(lhs);
	aml_mapper_visitor_destroy(rhs);
}

void test_mapper_visitor_size()
{
	size_t size;
	const size_t real_size =
	        sizeof(struct C) +
	        c->n * (sizeof(struct B) + sizeof(struct A) + sizeof(struct A));
	struct aml_mapper_visitor *it;

	aml_mapper_visitor_create(&it, (void *)c, &struct_C_mapper,
	                          aml_dma_linux, aml_dma_linux_memcpy_op);
	assert(aml_mapper_visitor_size(it, &size) == AML_SUCCESS);
	assert(size == real_size);
	aml_mapper_visitor_destroy(it);
}

int main(int argc, char **argv)
{
	// Init
	aml_init(&argc, &argv);
	init();

	// Tests
	test_mapper_visitor();
	test_mapper_visitor_size();
	test_mapper_visitor_match();

	// Cleanup
	teardown();
	aml_finalize();
	return 0;
}
