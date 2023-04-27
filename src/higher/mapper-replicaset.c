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

#include "aml/higher/mapper.h"
#include "aml/higher/mapper/creator.h"
#include "aml/higher/mapper/deepcopy.h"
#include "aml/higher/mapper/replicaset.h"
#include "aml/higher/mapper/visitor.h"

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
	out->num_shared = num_sharing;
	out->ptr = NULL;
	pthread_mutex_init(&out->producer, NULL);
	sem_init(&out->consumed, 0, 0);
	sem_init(&out->produced, 0, 0);
}

void aml_replica_build_fini(struct aml_shared_replica_config *out)
{
	assert(out != NULL);
	pthread_mutex_destroy(&out->producer);
	sem_destroy(&out->consumed);
	sem_destroy(&out->produced);
}

struct aml_replica_build_thread_args {
	aml_mapped_ptrs *replicaset_pointers;
	pthread_mutex_t *replicaset_mutex;
	pthread_mutex_t initialization_lock;
	pthread_cond_t initialization_cond;
	struct aml_mapper_creator *crtr;
	struct aml_shared_replica_config *local;
	struct aml_shared_replica_config *shared;
	struct aml_shared_replica_config *global;
};

static void *aml_mapper_replica_build_thread_fn(void *thread_args)
{
	int err, flags;
	const int flag_mask =
	        (AML_MAPPER_REPLICASET_LOCAL | AML_MAPPER_REPLICASET_SHARED |
	         AML_MAPPER_REPLICASET_GLOBAL) &
	        ~AML_MAPPER_FLAG_SPLIT;
	UT_array ptrs, crtrs;
	struct aml_mapper_creator **crtr_ptr, *next = NULL;

	// Pthread arguments local copy.
	struct aml_replica_build_thread_args args =
	        *(struct aml_replica_build_thread_args *)thread_args;
	struct aml_shared_replica_config *build = args.local;
	struct aml_mapped_ptr ptr = {.ptr = args.crtr->device_memory,
	                             .size = args.crtr->size,
	                             .area = args.local->area};
	unsigned int num_consumer = args.local->num_shared;

	// Signal parent thread that initialization is done and arguments on
	// stack will not be used anylonger.
	// However, pointers contained in args must remain valid for the
	// lifetime of this thread.
	pthread_mutex_lock(
	        &((struct aml_replica_build_thread_args *)thread_args)
	                 ->initialization_lock);
	pthread_cond_signal(
	        &((struct aml_replica_build_thread_args *)thread_args)
	                 ->initialization_cond);
	pthread_mutex_unlock(
	        &((struct aml_replica_build_thread_args *)thread_args)
	                 ->initialization_lock);

	// Allocate array of creators spawned in branches.
	utarray_init(&crtrs, &creator_icd);

	// Allocate and initialize array of device pointers allocated along
	// the way.
	utarray_init(&ptrs, &aml_replicaset_ptrs_icd);
	utarray_push_back(&ptrs, &ptr);

	// Byte copy current struct on host and move on to the element.
iterate_creator:
	err = aml_mapper_creator_next(args.crtr);

check_err:
	// The iteration or branch creation was successful and we can continue
	// iterating.
	if (err == AML_SUCCESS)
		goto iterate_creator;
	// The iteration or branch creation was successful but the next step of
	// iteration requires to branch.
	if (err == -AML_EINVAL)
		goto branch;
	// The iteration or branch creation was successful and the final
	// iteration step has been reached.
	if (err == -AML_EDOM)
		goto next_creator;
	assert(0);

branch:
	// Set the build configuration to what the mapper says.
	flags = args.crtr->stack->mapper->flags & flag_mask;
	if (flags & AML_MAPPER_REPLICASET_LOCAL) {
		build = args.local;
		num_consumer = args.local->num_shared;
	} else if (flags & AML_MAPPER_REPLICASET_SHARED) {
		build = args.shared;
		num_consumer = args.shared->num_shared;
	} else if (flags & AML_MAPPER_REPLICASET_GLOBAL) {
		build = args.global;
		// If this globally shared struct is a child of a shared struct,
		// then only producer threads of each shared struct will
		// synchronize to build this globally shared struct.
		if (num_consumer == args.shared->num_shared)
			num_consumer = args.global->num_shared /
			               args.shared->num_shared;
		// Only if the parent struct was private (local), we set
		// num_sharing to global. In other cases, the number of shared
		// copies was already global and with the appropriate value.
		else if (num_consumer == args.local->num_shared)
			num_consumer = args.global->num_shared;
	}

	// Check whether we will be the producer or a consumer.
	if (pthread_mutex_trylock(&build->producer) == 0)
		goto branch_producer;
	else
		goto branch_consumer;

branch_producer:
	// Initialize shared pointer to NULL. In case of failure, all threads
	// will see the NULL value and go to error flag.
	build->ptr = NULL;

	// Create the branch (allocation of device pointer).
	err = aml_mapper_creator_branch(&next, args.crtr, build->area,
	                                build->area_opts, build->dma_host_dst,
	                                build->memcpy_host_dst);

	// On success, update shared pointer and push the new creator in the
	// list of creator pending to be iterated.
	if (err == AML_SUCCESS || err == -AML_EDOM) {
		build->ptr = next->device_memory;
		utarray_push_back(&crtrs, &next);
	}

	// Signal the new pointer is ready to be consumed.
	for (unsigned i = 0; i < num_consumer - 1; i++)
		sem_post(&build->produced);
	// Wait for all consumers.
	for (unsigned i = 0; i < num_consumer - 1; i++)
		sem_wait(&build->consumed);
	// Release producer role.
	pthread_mutex_unlock(&build->producer);

	if (err != AML_SUCCESS && err != -AML_EDOM)
		goto error;
	else
		goto check_err;

branch_consumer:
	sem_wait(&build->produced);

	// On success, connect shared pointer to its parent struct.
	if (build->ptr != NULL) {
		err = aml_mapper_creator_connect(args.crtr, build->ptr);

		// Signal we consumed the shared pointer.
		sem_post(&build->consumed);

		// Success from the producer
		goto check_err;
	}

	// Signal we consumed the shared pointer.
	sem_post(&build->consumed);

	// Failure from the producer.
	err = -AML_FAILURE;
	goto error;

next_creator:
	// Save area in the device pointer of the successfully (host) copied
	// struct.
	ptr.area = args.crtr->device_area;
	// Finish struct deepcopy with a copy of the contiguous chunk on device.
	assert(aml_mapper_creator_finish(args.crtr, &ptr.ptr, &ptr.size) ==
	       AML_SUCCESS);
	// Append pointer to this part of the struct to the list of allocated
	// pointers for the overall structure.
	utarray_push_back(&ptrs, &ptr);
	// Fetch a new creator for another part of the structure.
	crtr_ptr = utarray_back(&crtrs);
	// If there is no more creator to fetch, the copy is finished.
	if (crtr_ptr == NULL)
		goto success;
	// Pop back last creator without calling destructor.
	crtrs.i--;
	args.crtr = *crtr_ptr;
	goto iterate_creator;

success:;
	// Get the pointer to the root of the structure.
	// This is what we return.
	struct aml_mapped_ptr *ret =
	        (struct aml_mapped_ptr *)utarray_front(&ptrs);
	void *out = ret->ptr;

	// Cleanup structures created in this function.
	utarray_done(&crtrs);
	// Append pointers created for this replica in the global array
	// of all replicas pointers.
	pthread_mutex_lock(args.replicaset_mutex);
	utarray_concat((UT_array *)args.replicaset_pointers, &ptrs);
	pthread_mutex_unlock(args.replicaset_mutex);
	// Cleanup the array container.
	utarray_done(&ptrs);
	return out;

error:
	// Cleanup currently built creator.
	aml_mapper_creator_abort(args.crtr);
	// Cleanup freshly obtained new creator.
	aml_mapper_creator_abort(next);
	// Cleanup created device pointers.
	aml_replicaset_ptrs_destroy(&ptrs);
	// Cleanup arrays
	utarray_done(&ptrs);
	// This cleans creators that have not been iteratated.
	utarray_done(&crtrs);
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

	// Since we have a visitor ready, we compute the allocation size here.
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
	        .initialization_lock = PTHREAD_MUTEX_INITIALIZER,
	        .initialization_cond = PTHREAD_COND_INITIALIZER,
	        .crtr = crtr,
	        .local = local,
	        .shared = shared,
	        .global = global,
	};
	pthread_mutex_lock(&args.initialization_lock);

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
	pthread_cond_wait(&args.initialization_cond, &args.initialization_lock);
	pthread_mutex_unlock(&args.initialization_lock);
	// FIXME helgrind complains if this is uncommented -- not sure why?!
	// pthread_mutex_destroy(&args.initialization_lock);
	pthread_cond_destroy(&args.initialization_cond);
	*thread_handle = thread;
	return AML_SUCCESS;
}
