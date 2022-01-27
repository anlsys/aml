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
#include "aml/higher/mapper/creator.h"
#include "aml/higher/mapper/visitor.h"

// Mapper args (linux)
#include "aml/area/linux.h"
#include "aml/dma/linux.h"

struct A {
	size_t dummy;
};
aml_final_mapper_decl(struct_A_mapper, 0, struct A);

struct B {
	struct A *first;
	struct A *second;
};
aml_mapper_decl(struct_B_mapper,
                AML_MAPPER_FLAG_SPLIT,
                struct B,
                first,
                &struct_A_mapper,
                second,
                &struct_A_mapper);

struct C {
	size_t n;
	struct B *first;
};

aml_mapper_decl(struct_C_mapper,
                AML_MAPPER_FLAG_HOST,
                struct C,
                first,
                n,
                &struct_B_mapper);

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
		a->dummy = i;
		c->first[i].first = a;
		c->first[i].second = a;
	}
}

int is_equal(struct C *other)
{
	// Scalars have the same value
	if (c->n != other->n)
		return 0;
	// Pointers have not been litterally copied.
	assert(c->first != other->first);

	// Arrays have the same number of elements.
	// Otherwise there will be a memory error on valgrind check.
	for (size_t i = 0; i < c->n; i++) {
		// Pointers have not been litterally copied.
		assert(c->first[i].first != other->first[i].first);
		assert(c->first[i].second != other->first[i].second);
		// Scalars have the same value
		if (c->first[i].first->dummy != other->first[i].first->dummy)
			return 0;
		if (c->first[i].second->dummy != other->first[i].second->dummy)
			return 0;
	}
	return 1;
}

void teardown()
{
	free(c);
}

void test_mapper_creator()
{
	struct aml_mapper_creator *creator, *branch;
	void *root, *split;
	size_t sizeof_root, sizeof_split;
	struct aml_mapper_visitor *lhs, *rhs;

	// Initialize creator
	assert(aml_mapper_creator_create(&creator, (void *)c, 0,
	                                 &struct_C_mapper, NULL, NULL, NULL,
	                                 NULL, NULL, NULL) == AML_SUCCESS);

	// Copy root.
	assert(aml_mapper_creator_next(creator) == AML_SUCCESS);
	// The next field has SPLIT flag on.
	assert(aml_mapper_creator_next(creator) == -AML_EINVAL);

	// Construct first field.
	assert(aml_mapper_creator_branch(&branch, creator, &aml_area_linux,
	                                 NULL, aml_dma_linux,
	                                 aml_dma_linux_memcpy_op) == -AML_EDOM);
	// Copy array.
	assert(aml_mapper_creator_next(branch) == AML_SUCCESS);
	// Array element 0, first field.
	assert(aml_mapper_creator_next(branch) == AML_SUCCESS);
	// Array element 0, second field.
	assert(aml_mapper_creator_next(branch) == AML_SUCCESS);
	// Array element 1, first field.
	assert(aml_mapper_creator_next(branch) == AML_SUCCESS);
	// Array element 1, second field.
	// This is the last thing to copy.
	assert(aml_mapper_creator_next(branch) == -AML_EDOM);
	assert(aml_mapper_creator_finish(branch, &split, &sizeof_split) ==
	       AML_SUCCESS);
	assert(sizeof_split ==
	       c->n * (sizeof(struct B) + sizeof(struct A) + sizeof(struct A)));

	// Back to root copy.
	// There should be nothing left to copy.
	assert(aml_mapper_creator_next(creator) == -AML_EDOM);
	// Done.
	assert(aml_mapper_creator_finish(creator, &root, &sizeof_root) ==
	       AML_SUCCESS);
	assert(sizeof_root == sizeof(struct C));

	// Check the copied structure and original structure are identical.
	// is_equal() test adds to the aml_mapper_visitor_match() test that it
	// makes sure fields pointer are different, i.e they have not been
	// literally copied.
	assert(is_equal(root));
	aml_mapper_visitor_create(&lhs, (void *)c, &struct_C_mapper,
	                          aml_dma_linux, aml_dma_linux_memcpy_op);
	aml_mapper_visitor_create(&rhs, (void *)root, &struct_C_mapper, NULL,
	                          NULL);
	assert(aml_mapper_visitor_match(lhs, rhs) == 1);
	// Try to change original structure to break the match.
	c->first[0].first->dummy = 32456;
	assert(aml_mapper_visitor_match(lhs, rhs) == 0);
	c->first[0].first->dummy = 0;
	c->n = 1;
	assert(aml_mapper_visitor_match(lhs, rhs) == 0);
	// Restore to the same structure.
	c->n = 2;
	assert(aml_mapper_visitor_match(lhs, rhs) == 1);

	// Cleanup
	aml_mapper_visitor_destroy(lhs);
	aml_mapper_visitor_destroy(rhs);
	free(root);
	aml_area_munmap(&aml_area_linux, split, sizeof_split);
}

int main(int argc, char **argv)
{
	// Init
	aml_init(&argc, &argv);
	init();

	// Tests
	test_mapper_creator();

	// Cleanup
	teardown();
	aml_finalize();
	return 0;
}
