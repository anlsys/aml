#include <pthread.h>
#include <semaphore.h>

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

int aml_replica_build_init(struct aml_shared_replica_config *out,
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
	out->num_shared = num_sharing;
	sem_init(&out->ptr_ready, 0, num_sharing);	
	sem_post(&out->ptr_ready);

	return AML_SUCCESS;
}

int aml_replica_build_fini(struct aml_replica_build *out)
{
	assert(out != NULL);
	sem_destroy(&out->ptr_ready);
}

struct aml_replica_build_thread_args {
	aml_mapped_ptrs *replicaset_pointers;
	pthread_mutex_t *replicaset_mutex;
	struct aml_mapper_creator *crtr;
	struct aml_shared_replica_config *local;
	struct aml_shared_replica_config *shared;
	struct aml_shared_replica_config *global;
	pthread_mutex_t initialization_lock;
};

static void *aml_mapper_replica_build_thread_fn(void *thread_args)
{
	int err;
	UT_array ptrs, crtrs;
	struct aml_mapper_creator **crtr_ptr, *next = NULL;
	struct aml_mapped_ptr ptr = {.ptr = NULL, .size = 0, .area = NULL};

	// Pthread arguments local copy.
	struct aml_replica_build_thread_args *args =
	        (struct aml_replica_build_thread_args *)thread_args;
	struct aml_shared_replica_config *build = args->local;
	aml_mapped_ptrs *replicaset_pointers = args->replicaset_pointers;
	pthread_mutex_t *replicaset_mutex = args->replicaset_mutex;
	struct aml_mapper_creator *crtr = args->crtr;
	struct aml_shared_replica_config *local = args->local;
	struct aml_shared_replica_config *shared = args->shared;
	struct aml_shared_replica_config *global = args->global;
	// From this point we don't access args anylonger.
	// However, pointers contained in args must remain valid for the
	// lifetime of this thread.
	pthread_mutex_unlock(&args->initialization_lock);

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
		build = args->local;
	else if (crtr->stack->mapper->flags & AML_MAPPER_REPLICASET_SHARED)
		build = args->shared;
	else if (crtr->stack->mapper->flags & AML_MAPPER_REPLICASET_GLOBAL)
		build = args->global;

	// Make a branch.
	// If the thread gets the lock, it is responsible for allocation.
	// The other threads wait for allocation to finish and connect the
	// pointer allocated by the former thread to their own replica.
	if (sem_trywait(&build->ptr_ready) == 0) {
		build->ptr = NULL;
		err = aml_mapper_creator_branch(&next, build->area,
		                                build->area_opts,
		                                build->dma_host_dst,
		                                build->memcpy_host_dst);

		if (err == AML_SUCCESS || err == -AML_EDOM)
			build->ptr = next->device_memory;
		for (size_t i = 0; i < build->num_sharing; i++)
			sem_post(&build->ptr_ready);
		if (err != AML_SUCCESS && err != -AML_EDOM)
			goto error_with_barrier;

		// Push new creator to be in the stack of pending creators.
		utarray_push_back(&crtrs, &next);
	} else {
		sem_wait(&build->ptr_ready);
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
	       "Unexpected failure from a replica build while connecting a "
	       "pointer allocated by another thread.");

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
	return (void *)AML_SUCCESS;

error_with_barrier:
	sem_trywait(&build->ptr_ready);
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
	return (void *)err;
}

int aml_mapper_replica_build_start(pthread_t *thread_handle,
                                   struct aml_mapper_visitor *visitor,
                                   struct aml_shared_replica_config *local,
                                   struct aml_shared_replica_config *shared,
                                   struct aml_shared_replica_config *global,
                                   aml_mapped_ptrs *replicaset_pointers,
                                   pthread_mutex_t *replicaset_mutex)
{
	struct aml_mapper_creator *crtr;
	size_t size;

	// Since we have a visitor ready, we compute size here.
	int err = aml_mapper_visitor_size(visitor, &size);
	if (err != AML_SUCCESS)
		return err;

	// Make the creator for the local replica starting where visitor is
	// pointing.
	err = aml_mapper_creator_create(&crtr, visitor->stack->device_ptr, size,
	                                visitor->stack->mapper, local->area,
	                                local->area_opts, visitor->dma,
	                                local->dma_host_dst, visitor->memcpy_op,
	                                local->memcpy_host_dst);
	if (err != AML_SUCCESS)
		return err;

	struct aml_replica_build_thread_args args = {
	        .replicaset_pointers = replicaset_pointers,
	        .replicaset_mutex = replicaset_mutex,
	        .crtr = crtr,
	        .local = local,
	        .shared = shared,
	        .global = global,
	};
	pthread_mutex_init(&args.initialization_lock);

	// Start the thread responsible to make this replica.
	// It is important that all threads responsible for all replicas
	// get started, otherwise, threads will likely end up in a deadlock
	// from waiting on shared copies in the ptr_ready semaphore.
	pthread_t thread;
	switch (pthread_create(&thread, NULL, (void *)&args)) {
	EAGAIN:
		aml_mapper_creator_abort(crtr);
		return -AML_FAILURE;
	EINVAL:
	EPERM:
		assert(0 && "Unexpected error with NULL pthread attribute from "
		            "AML.");
	default:
		break;
	}

	// Wait thread initialization before args (on stack) is destroyed.
	pthread_mutex_lock(&args.initialization_lock);
	pthread_mutex_unlock(&args.initialization_lock);
	*thread_handle = thread;
	return AML_SUCCESS;
}
