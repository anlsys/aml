#include <pthread.h>
#include <semaphore.h>

#include "aml.h"

#include "aml/higher/mapper.h"
#include "aml/higher/mapper/creator.h"
#include "aml/higher/mapper/deepcopy.h"
#include "aml/higher/mapper/visitor.h"
#include "aml/higher/mapper/replicaset.h"

#include "internal/utarray.h"

static UT_icd aml_replicaset_ptrs_icd = {
        .sz = sizeof(struct aml_mapped_ptr),
        .init = NULL,
        .copy = NULL,
        .dtor = NULL,
};

static void aml_mapper_creator_destroy(void *elt)
{
	struct aml_mapper_creator *c = *(struct aml_mapper_creator **)elt;
	(void)aml_mapper_creator_abort(c);
}

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

void aml_replica_build_init(struct aml_shared_replica_config *out,
                            unsigned num_sharing,
                            struct aml_area *area,
                            struct aml_area_mmap_options *opts,
                            struct aml_dma *dma_host_dst,
                            aml_dma_operator memcpy_host_dst)
{
	assert(out != NULL);
	out->area = area;
	out->area_opts = opts;
	out->dma_host_dst = dma_host_dst;
	out->memcpy_host_dst = memcpy_host_dst;
	out->ptr = NULL;
	pthread_mutex_init(&out->lock, NULL);
	pthread_barrier_init(&out->barrier, NULL, num_sharing);
}

void aml_replica_build_fini(struct aml_shared_replica_config *out)
{
	assert(out != NULL);
	pthread_barrier_destroy(&out->barrier);
	pthread_mutex_destroy(&out->lock);
}

struct aml_replica_build_thread_args {
	aml_mapped_ptrs *replicaset_pointers;
	pthread_mutex_t *replicaset_mutex;
	struct aml_mapper_creator *crtr;
	struct aml_shared_replica_config *local;
	struct aml_shared_replica_config *shared;
	struct aml_shared_replica_config *global;
};

static void *aml_mapper_replica_build_thread_fn(void *thread_args)
{
	int err, flags;
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
	pthread_mutex_unlock(&local->lock);

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
	flags = crtr->stack->mapper->flags & 0xe0000000;
	if (flags & AML_MAPPER_REPLICASET_LOCAL)
		build = local;
	else if (flags & AML_MAPPER_REPLICASET_SHARED)
		build = shared;
	else if (flags & AML_MAPPER_REPLICASET_GLOBAL)
		build = global;

	// Make a branch.
	// If the thread gets the lock, it is responsible for allocation.
	// The other threads wait for allocation to finish and connect the
	// pointer allocated by the former thread to their own replica.
	if (pthread_mutex_trylock(&build->lock) == 0) {
		build->ptr = NULL;
		err = aml_mapper_creator_branch(&next, crtr, build->area,
		                                build->area_opts,
		                                build->dma_host_dst,
		                                build->memcpy_host_dst);

		if (err == AML_SUCCESS || err == -AML_EDOM)
			build->ptr = next->device_memory;
		pthread_barrier_wait(&build->barrier);
		pthread_mutex_unlock(&build->lock);
		if (err != AML_SUCCESS && err != -AML_EDOM)
			goto error;

		// Push new creator to be in the stack of pending creators.
		utarray_push_back(&crtrs, &next);
	} else {
		pthread_barrier_wait(&build->barrier);
		fprintf(stderr, "Shallow copy.\n");
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
	utarray_push_back(&ptrs, &ptr);
	crtr_ptr = utarray_back(&crtrs);
	if (crtr_ptr == NULL)
		goto success;
	crtrs.i--; // Pop back without calling destructor.
	crtr = *crtr_ptr;
	goto iterate_creator;

success:;
	struct aml_mapped_ptr *ret =
	        (struct aml_mapped_ptr *)utarray_front(&ptrs);
	void *out = ret->ptr;

	utarray_done(&crtrs);
	pthread_mutex_lock(replicaset_mutex);
	utarray_concat((UT_array *)replicaset_pointers, &ptrs);
	pthread_mutex_unlock(replicaset_mutex);
	utarray_done(&ptrs);
	return out;

error:
	if (crtr != NULL)
		aml_mapper_creator_abort(crtr);
	if (next != NULL)
		aml_mapper_creator_abort(next);
	aml_replicaset_ptrs_destroy(&ptrs);
	utarray_done(&ptrs);
	utarray_done(&crtrs);
	aml_replica_build_fini(local);
	aml_replica_build_fini(shared);
	aml_replica_build_fini(global);
	aml_errno = err;
	return NULL;
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
	pthread_mutex_lock(&local->lock);

	// Start the thread responsible to make this replica.
	// It is important that all threads responsible for all replicas
	// get started, otherwise, threads will likely end up in a deadlock
	// from waiting on shared copies in the ptr_ready semaphore.
	pthread_t thread;
	switch (pthread_create(&thread, NULL,
	                       aml_mapper_replica_build_thread_fn,
	                       (void *)&args)) {
	case EAGAIN:
		aml_mapper_creator_abort(crtr);
		return -AML_FAILURE;
	case EINVAL:
	case EPERM:
		assert(0 && "Unexpected error with NULL pthread attribute from "
		            "AML.");
	default:
		break;
	}

	// Wait thread initialization before args (on stack) is destroyed.
	pthread_mutex_lock(&local->lock);
	pthread_mutex_unlock(&local->lock);
	*thread_handle = thread;
	return AML_SUCCESS;
}
