/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include <pthread.h>
#include <semaphore.h>

#include "aml.h"

#include "aml/area/linux.h"
#include "aml/dma/linux.h"
#include "aml/higher/mapper.h"
#include "aml/higher/mapper/creator.h"
#include "aml/higher/mapper/deepcopy.h"
#include "aml/higher/mapper/replicaset.h"
#include "aml/higher/mapper/visitor.h"

#include "internal/utarray.h"

//------------------------------------------------------------------------------
// Declaration of the structures to copy.
//------------------------------------------------------------------------------

struct A {
	size_t dummy;
};
aml_final_mapper_decl(struct_A_mapper, AML_MAPPER_REPLICASET_GLOBAL, struct A);

struct B {
	struct A *a;
};
aml_mapper_decl(struct_B_mapper,
                AML_MAPPER_REPLICASET_SHARED,
                struct B,
                a,
                &struct_A_mapper);

struct C {
	size_t n_b; // 2
	struct B *b;
	size_t n_a; // 1
	struct A *a;
};

aml_mapper_decl(struct_C_mapper,
                AML_MAPPER_REPLICASET_LOCAL,
                struct C,
                a,
                n_a,
                &struct_A_mapper,
                b,
                n_b,
                &struct_B_mapper);

struct aml_shared_replica_config global;
struct aml_shared_replica_config shared[2];
struct aml_shared_replica_config local[4];
UT_array *replicaset_pointers;
pthread_mutex_t replicaset_mutex;
struct aml_mapper_visitor *visitor;
struct C *c;
struct C *replicas[4];

int is_equal(struct C *other)
{
	// Scalars have the same value
	if (c->n_a != other->n_a || c->n_b != other->n_b)
		return 0;
	if (c->a->dummy != other->a->dummy)
		return 0;

	// Arrays have the same number of elements.
	// Otherwise there will be a memory error on valgrind check.
	for (size_t i = 0; i < c->n_b; i++) {
		// Scalars have the same value
		if (c->b[i].a->dummy != other->b[i].a->dummy)
			return 0;
	}

	return 1;
}

static UT_icd aml_mapped_ptr_icd = {
        .sz = sizeof(struct aml_mapped_ptr),
        .init = NULL,
        .copy = NULL,
        .dtor = aml_mapped_ptr_destroy,
};

//------------------------------------------------------------------------------
// Test.
//------------------------------------------------------------------------------

void test_replicaset()
{
	const size_t n = 4;
	pthread_t threads[n];

	// Build replicas
	for (size_t i = 0; i < n; i++)
		assert(aml_mapper_replica_build_start(
		               threads + i, visitor, local + i,
		               shared + (i / 2), &global,
		               (aml_mapped_ptrs *)replicaset_pointers,
		               &replicaset_mutex) == AML_SUCCESS);
	for (size_t i = 0; i < n; i++) {
		pthread_join(threads[i], (void *)(replicas + i));
		assert(replicas[i] != NULL);
	}

	// Test equality.
	for (size_t i = 0; i < n; i++)
		assert(is_equal(replicas[i]));

	// Test sharing.
	assert(replicas[0]->a == replicas[1]->a);
	assert(replicas[0]->a == replicas[2]->a);
	assert(replicas[0]->a == replicas[3]->a);

	assert(replicas[0]->b == replicas[1]->b);
	assert(replicas[2]->b == replicas[3]->b);

	assert(replicas[0]->b[0].a == replicas[1]->b[0].a);
	assert(replicas[0]->b[0].a == replicas[2]->b[0].a);
	assert(replicas[0]->b[0].a == replicas[3]->b[0].a);
	assert(replicas[0]->b[1].a == replicas[1]->b[1].a);
	assert(replicas[0]->b[1].a == replicas[2]->b[1].a);
	assert(replicas[0]->b[1].a == replicas[3]->b[1].a);
}

void init()
{
	// Initialize C structure.
	c = malloc(sizeof(*c));
	assert(c != NULL);

	c->n_a = 1;
	c->a = malloc(c->n_a * sizeof(struct A));
	assert(c->a != NULL);
	c->a->dummy = 2;

	c->n_b = 2;
	c->b = malloc(c->n_b * sizeof(struct B));
	assert(c->b != NULL);

	c->b[0].a = malloc(sizeof(struct A));
	assert(c->b[0].a != NULL);
	c->b[0].a->dummy = 0;

	c->b[1].a = malloc(sizeof(struct A));
	assert(c->b[1].a != NULL);
	c->b[1].a->dummy = 1;

	// Initialize build config.
	aml_replica_build_init(&global, 4, &aml_area_linux, NULL, aml_dma_linux,
	                       aml_dma_linux_memcpy_op);

	for (unsigned i = 0; i < 2; i++) {
		aml_replica_build_init(shared + i, 2, &aml_area_linux, NULL,
		                       aml_dma_linux, aml_dma_linux_memcpy_op);
	}

	for (unsigned i = 0; i < 4; i++) {
		aml_replica_build_init(local + i, 1, &aml_area_linux, NULL,
		                       aml_dma_linux, aml_dma_linux_memcpy_op);
	}

	// Build other global variables
	utarray_new(replicaset_pointers, &aml_mapped_ptr_icd);
	pthread_mutex_init(&replicaset_mutex, NULL);
	assert(aml_mapper_visitor_create(
	               &visitor, (void *)c, &struct_C_mapper, aml_dma_linux,
	               aml_dma_linux_memcpy_op) == AML_SUCCESS);
}

void teardown()
{
	assert(aml_mapper_visitor_destroy(visitor) == AML_SUCCESS);
	pthread_mutex_destroy(&replicaset_mutex);
	aml_mapper_deepfree(replicaset_pointers);

	// Free replicas
	aml_replica_build_fini(&global);
	for (unsigned i = 0; i < sizeof(shared) / sizeof(*shared); i++)
		aml_replica_build_fini(shared + i);
	for (unsigned i = 0; i < sizeof(local) / sizeof(*local); i++)
		aml_replica_build_fini(local + i);

	// Free C structure.
	free(c->a);
	free(c->b[0].a);
	free(c->b[1].a);
	free(c->b);
	free(c);
}

int main(int argc, char **argv)
{
	assert(aml_init(&argc, &argv) == AML_SUCCESS);
	init();

	test_replicaset();

	teardown();
	aml_finalize();
}
