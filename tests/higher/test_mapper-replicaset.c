#include <pthread.h>
#include <semaphore.h>

#include "aml.h"

#include "aml/area/linux.h"
#include "aml/dma/linux.h"
#include "aml/higher/mapper.h"
#include "aml/higher/mapper/creator.h"
#include "aml/higher/mapper/deepcopy.h"
#include "aml/higher/mapper/visitor.h"
#include "aml/higher/mapper/replicaset.h"

#include "internal/utarray.h"

struct A {
	size_t dummy;
};
aml_final_mapper_decl(struct_A_mapper, AML_MAPPER_REPLICASET_GLOBAL, struct A);

struct B {
	struct A *first;
};
aml_mapper_decl(struct_B_mapper,
                AML_MAPPER_REPLICASET_SHARED,
                struct B,
                first,
                &struct_A_mapper);

struct C {
	size_t n;
	struct B *first;
};

aml_mapper_decl(struct_C_mapper,
                AML_MAPPER_REPLICASET_LOCAL,
                struct C,
                first,
                n,
                &struct_B_mapper);

struct aml_shared_replica_config global;
struct aml_shared_replica_config shared[2];
struct aml_shared_replica_config local[4];
struct C *c;

void init()
{
	// Initialize C structure.
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
	}

	// Initialize build config.
	aml_replica_build_init(&global, 4,
	                       &aml_area_linux, NULL, aml_dma_linux,
	                       aml_dma_linux_memcpy_op);

	for (unsigned i = 0; i < 2; i++) {
		aml_replica_build_init(shared + i, 2,
		                       &aml_area_linux, NULL, aml_dma_linux,
		                       aml_dma_linux_memcpy_op);
	}

	for (unsigned i = 0; i < 4; i++) {
		aml_replica_build_init(local + i, 1, &aml_area_linux, NULL,
		                       aml_dma_linux, aml_dma_linux_memcpy_op);
	}
}

void teardown()
{
	aml_replica_build_fini(&global);
	for (unsigned i = 0; i < sizeof(shared) / sizeof(*shared); i++)
		aml_replica_build_fini(shared + i);
	for (unsigned i = 0; i < sizeof(local) / sizeof(*local); i++)
		aml_replica_build_fini(local + i);
	free(c);
}

int is_equal(struct C *other)
{
	// Scalars have the same value
	if (c->n != other->n)
		return 0;
	// Arrays have the same number of elements.
	// Otherwise there will be a memory error on valgrind check.
	for (size_t i = 0; i < c->n; i++) {
		// Scalars have the same value
		if (c->first[i].first->dummy != other->first[i].first->dummy)
			return 0;
	}
	return 1;
}

#define C_SHARED 1
#define B_SHARED (1 << 1)
#define A_SHARED (1 << 2)

int get_sharing(struct C *a, struct C *b)
{
	if (a == b)
		return C_SHARED | B_SHARED | A_SHARED;
	if (a->first == b->first)
		return B_SHARED | A_SHARED;
	for (size_t i = 0; i < a->n; i++)
		if (a->first[i].first != b->first[i].first)
			return 0;
	return A_SHARED;
}

static UT_icd aml_mapped_ptr_icd = {
        .sz = sizeof(struct aml_mapped_ptr),
        .init = NULL,
        .copy = NULL,
        .dtor = aml_mapped_ptr_destroy,
};

void test_replicaset()
{
	const size_t n = 4;
	pthread_t threads[n];
	void *replicas[4];
	UT_array *replicaset_pointers;
	pthread_mutex_t replicaset_mutex;
	struct aml_mapper_visitor *visitor;

	utarray_new(replicaset_pointers, &aml_mapped_ptr_icd);
	pthread_mutex_init(&replicaset_mutex, NULL);
	assert(aml_mapper_visitor_create(
	               &visitor, (void *)c, &struct_C_mapper, aml_dma_linux,
	               aml_dma_linux_memcpy_op) == AML_SUCCESS);

	// Build replicas
	for (size_t i = 0; i < n; i++)
		assert(aml_mapper_replica_build_start(
		               threads + i, visitor,
									 local + i,
		               shared + (i / 2),
									 &global,
		               (aml_mapped_ptrs *)replicaset_pointers,
		               &replicaset_mutex) == AML_SUCCESS);
	for (size_t i = 0; i < n; i++) {
		pthread_join(threads[i], replicas + i);
		assert(replicas[i] != NULL);
	}

	// Test equality.
	for (size_t i = 0; i < n; i++)
		assert(is_equal(replicas[i]));

	// Test sharing.
	assert(get_sharing((struct C *)replicas[0], (struct C *)replicas[1]) ==
	       (B_SHARED | A_SHARED));
	assert(get_sharing((struct C *)replicas[2], (struct C *)replicas[3]) ==
	       (B_SHARED | A_SHARED));
	assert(get_sharing((struct C *)replicas[0], (struct C *)replicas[2]) ==
	       A_SHARED);
	assert(get_sharing((struct C *)replicas[0], (struct C *)replicas[3]) ==
	       A_SHARED);
	assert(get_sharing((struct C *)replicas[1], (struct C *)replicas[2]) ==
	       A_SHARED);
	assert(get_sharing((struct C *)replicas[1], (struct C *)replicas[3]) ==
	       A_SHARED);

	// Cleanup
	assert(aml_mapper_visitor_destroy(visitor) == AML_SUCCESS);
	pthread_mutex_destroy(&replicaset_mutex);
	utarray_free(replicaset_pointers);
}

int main(int argc, char **argv)
{
	assert(aml_init(&argc, &argv) == AML_SUCCESS);
	init();

	test_replicaset();

	teardown();
	aml_finalize();
}
