#include <pthread.h>

#include "aml/higher/mapper/creator.h"
#include "aml/higher/mapper/replicaset.h"

#include "internal/utarray.h"

static UT_icd aml_replicaset_ptrs_icd = {
        .sz = sizeof(struct aml_replicaset_ptr),
        .init = NULL,
        .copy = NULL,
        .dtor = NULL,
};

static UT_icd creator_icd = {
        .sz = sizeof(struct aml_mapper_creator *),
        .init = NULL,
        .copy = NULL,
        .dtor = aml_mapper_creator_destroy,
};

static void aml_replicaset_ptrs_destroy(UT_array *ptrs)
{
	void *ptr;
	while ((ptr = utarray_back(ptrs)) != NULL) {
		aml_mapped_ptr_destroy(ptr);
		utarray_pop_back(ptrs);
	}
}

int aml_replica_build_init(struct aml_replica_build *out,
                           unsigned num_sharing,
                           struct aml_area *area,
                           struct aml_area_mmap_options *opts,
                           struct aml_dma *dma_host_dst,
                           aml_dma_operator memcpy_host_dst)
{
	int err;

	assert(out != NULL);
	out->area = area;
	out->area_opts = opts;
	out->dma_host_dst = dma_host_dst;
	out->memcpy_host_dst = memcpy_host_dst;
	out->ptr = NULL;

	pthread_mutex_init(&out->mutex, NULL);
	err = pthread_barrier_init(&out->barrier, NULL, num_sharing);
	switch (err) {
	case EINVAL:
		return -AML_EINVAL;
	case ENOMEM:
		return -AML_ENOMEM;
	case EAGAIN:
		return -AML_FAILURE;
	default:
		break;
	}

	return AML_SUCCESS;
}

static int aml_replica_build_fini(struct aml_replica_build *out)
{
	assert(out != NULL);
	pthread_barrier_destroy(&out->barrier);
	pthread_mutex_destroy(&out->mutex);
}

int aml_mapper_replica_build(aml_mapped_ptrs *replicaset_pointers,
                             pthread_mutex_t *replicaset_mutex,
                             struct aml_mapper_creator *crtr,
                             struct aml_replica_build *local,
                             struct aml_replica_build *shared,
                             struct aml_replica_build *global)
{
	int err;
	UT_array ptrs, crtrs;
	struct aml_mapper_creator **crtr_ptr, *next = NULL;
	struct aml_mapped_ptr ptr = {.ptr = NULL, .size = 0, .area = area};
	struct aml_replica_build *build = local;

	// Allocate array of creators spawned in branches.
	utarray_init(&crtrs, &creator_icd);

	// Allocate and initialize pointer array.
	utarray_init(&ptrs, &aml_replicaset_ptrs_icd);

	// Start copy.
iterate_creator:
	err = aml_mapper_creator_next(crtr);
	if (err == AML_SUCCESS)
		goto iterate_creator;
	if (err == -AML_EINVAL)
		goto branch;
	if (err == -AML_EDOM)
		goto next_creator;
	assert(0 &&
	       "Unexpected failure from a replica build while iterating a creator.");

branch:
	// Set the build configuration to what the mapper says.
	// If the flag AML_MAPPER_FLAG_SPLIT is met but no other replicaset
	// flags is set, the build configuration remains the same.
	if (crtr->stack->mapper->flags & AML_MAPPER_REPLICASET_LOCAL)
		build = local;
	else if (crtr->stack->mapper->flags & AML_MAPPER_REPLICASET_SHARED)
		build = shared;
	else if (crtr->stack->mapper->flags & AML_MAPPER_REPLICASET_GLOBAL)
		build = global;

	// Make a branch.
	// If the thread gets the lock, it is responsible for allocation.
	// The other threads wait for allocation to finish and connect the
	// pointer allocated by the former thread to their replica.
	if (pthread_mutex_trylock(&build->mutex) == 0) {
		err = aml_mapper_creator_branch(&next, build->area,
		                                build->area_opts,
		                                build->dma_host_dst,
		                                build->memcpy_host_dst);
		if (err != AML_SUCCESS && err != -AML_EDOM)
			goto error_with_barrier;
		build->ptr = next->device_memory;
		pthread_barrier_wait(build->barrier);
		// Push new creator to be in the stack of pending creators.
		utarray_push_back(&crtrs, &next);
	} else {
		pthread_barrier_wait(build->barrier);
		if (build->ptr == NULL) {
			err = -AML_FAILURE;
			goto error;
		}
		err = aml_mapper_creator_connect(crtr, build->ptr);
	}
	if (err == AML_SUCCESS)
		goto iterate_creator;
	if (err == -AML_EINVAL)
		goto branch;
	if (err == -AML_EDOM)
		goto next_creator;
	assert(0 &&
	       "Unexpected failure from a replica build while connecting a pointer allocated by another thread.");

next_creator:
	assert(aml_mapper_creator_finish(crtr, &ptr.ptr, &ptr.size) ==
	       AML_SUCCESS);
	utarray_push_back(ptrs, &ptr);
	crtr_ptr = utarray_back(&crtrs);
	if (crtr_ptr == NULL)
		goto success;
	crtrs.i--; // Pop back without calling destructor.
	crtr = *crtr_ptr;
	goto iterate_creator;

success:
	utarray_done(&crtrs);
	pthread_mutex_lock(replicaset_mutex);
	utarray_concat(replicaset_pointers, ptrs);
	pthread_mutex_unlock(replicaset_mutex);
	utarray_done(&ptrs);
	aml_replica_build_fini(local);
	aml_replica_build_fini(shared);
	aml_replica_build_fini(global);
	return AML_SUCCESS;

error_with_barrier:
	build->ptr = NULL;
	pthread_mutex_unlock(build->mutex);
	pthread_barrier_wait(build->barrier);
error:
	if (crtr != NULL)
		aml_mapper_creator_abort(crtr);
	if (next != NULL)
		aml_mapper_creator_abort(next);
	if (ptrs != NULL) {
		aml_replicaset_ptrs_destroy(ptrs);
		utarray_done(&ptrs);
	}
	if (crtrs != NULL)
		utarray_done(&crtrs);
	aml_replica_build_fini(local);
	aml_replica_build_fini(shared);
	aml_replica_build_fini(global);
	return err;
}
