/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include "aml.h"

#include "aml/area/hwloc.h"
#include "aml/higher/replicaset.h"
#include "aml/higher/replicaset/hwloc.h"

extern hwloc_topology_t aml_topology;

int aml_replicaset_hwloc_alloc(struct aml_replicaset **out,
                               const hwloc_obj_type_t initiator_type)
{
	struct aml_replicaset *replicaset = NULL;
	struct aml_replicaset_hwloc_data *data = NULL;

	// Check initiator type.
	const unsigned int n_initiator =
	        hwloc_get_nbobjs_by_type(aml_topology, initiator_type);
	hwloc_obj_t initiator =
	        hwloc_get_obj_by_type(aml_topology, initiator_type, 0);
	if (n_initiator == 0)
		return -AML_EDOM;
	if (initiator == NULL || initiator->cpuset == NULL ||
	    hwloc_bitmap_weight(initiator->cpuset) <= 0)
		return -AML_EINVAL;

	const unsigned int n_numa =
	        hwloc_get_nbobjs_by_type(aml_topology, HWLOC_OBJ_NUMANODE);

	// Allocation
	replicaset = AML_INNER_MALLOC_ARRAY(n_numa + n_initiator, void *,
	                                    struct aml_replicaset,
	                                    struct aml_replicaset_hwloc_data);
	if (replicaset == NULL)
		return -AML_ENOMEM;

	// Set ops
	replicaset->ops = &aml_replicaset_hwloc_ops;

	// Set data
	replicaset->data =
	        (struct aml_replicaset_data *)AML_INNER_MALLOC_GET_FIELD(
	                replicaset, 2, struct aml_replicaset,
	                struct aml_replicaset_hwloc_data);
	data = (struct aml_replicaset_hwloc_data *)replicaset->data;

	// Set replica pointers array
	replicaset->replica = (void **)AML_INNER_MALLOC_GET_ARRAY(
	        replicaset, void *, struct aml_replicaset,
	        struct aml_replicaset_hwloc_data);
	for (unsigned i = 0; i < n_numa; i++)
		replicaset->replica[i] = NULL;

	// Set initiator pointers array
	data->ptr = replicaset->replica + n_numa;

	// Set number of initiators
	data->num_ptr = n_initiator;

	// Set number of replicas to 0. Initialization will set
	// it to the correct value.
	replicaset->n = 0;

	*out = replicaset;
	return AML_SUCCESS;
}

int aml_replicaset_hwloc_create(struct aml_replicaset **out,
                                const size_t size,
                                const hwloc_obj_type_t initiator_type,
                                const enum hwloc_distances_kind_e kind)
{
	int err = -AML_FAILURE;
	struct aml_replicaset *replicaset = NULL;
	struct aml_replicaset_hwloc_data *data = NULL;
	struct aml_area *area = NULL;
	struct aml_area_hwloc_preferred_data *area_data = NULL;
	const unsigned int n_numa =
	        hwloc_get_nbobjs_by_type(aml_topology, HWLOC_OBJ_NUMANODE);
	hwloc_obj_t targets[n_numa];

	err = aml_replicaset_hwloc_alloc(&replicaset, initiator_type);
	if (err != AML_SUCCESS)
		return err;
	replicaset->size = size;
	data = (struct aml_replicaset_hwloc_data *)replicaset->data;

	// For each initiator allocate replica on preferred area
	for (hwloc_obj_t initiator =
	             hwloc_get_obj_by_type(aml_topology, initiator_type, 0);
	     initiator != NULL; initiator = initiator->next_cousin) {

		// Get preferred area.
		err = aml_area_hwloc_preferred_create(&area, initiator, kind);
		if (err != AML_SUCCESS)
			goto err_with_replicaset;
		area_data = (struct aml_area_hwloc_preferred_data *)area->data;

		// Search if preferred numa node is already a target
		for (unsigned i = 0; i < replicaset->n; i++) {
			if (targets[i] == area_data->numanodes[0]) {
				data->ptr[initiator->logical_index] =
				        replicaset->replica[i];
				goto next;
			}
		}

		// Preferred numa node is not a target yet.
		void *ptr = aml_area_mmap(area, size, NULL);
		if (ptr == NULL) {
			err = -AML_ENOMEM;
			goto err_with_replicas;
		}

		replicaset->replica[replicaset->n] = ptr;
		data->ptr[initiator->logical_index] = ptr;
		targets[replicaset->n] = area_data->numanodes[0];
		replicaset->n++;

	next:
		// Area cleanup
		aml_area_hwloc_preferred_destroy(&area);
	}

	// Success
	*out = replicaset;
	return AML_SUCCESS;

	// Failure
err_with_replicas:
	for (unsigned i = 0; i < replicaset->n; i++)
		munmap(replicaset->replica[i], size);

err_with_replicaset:
	free(replicaset);
	return err;
}

void aml_replicaset_hwloc_destroy(struct aml_replicaset **replicaset)
{
	if (replicaset == NULL || *replicaset == NULL)
		return;

	for (unsigned int i = 0; i < (*replicaset)->n; i++)
		munmap((*replicaset)->replica[i], (*replicaset)->size);
	free(*replicaset);
	*replicaset = NULL;
}

int aml_replicaset_hwloc_init(struct aml_replicaset *replicaset,
                              const void *data)
{
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (unsigned i = 0; i < replicaset->n; i++)
		memcpy(replicaset->replica[i], data, replicaset->size);
	return AML_SUCCESS;
}

int aml_replicaset_hwloc_sync(struct aml_replicaset *replicaset,
                              const unsigned int id)
{
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (unsigned i = 0; i < replicaset->n; i++)
		if (i != id)
			memcpy(replicaset->replica[i], replicaset->replica[id],
			       replicaset->size);
	return AML_SUCCESS;
}

// See src/area/hwloc.c
int aml_hwloc_local_initiator(hwloc_obj_t *out);

void *aml_replicaset_hwloc_local_replica(struct aml_replicaset *replicaset)
{
	int err;
	hwloc_obj_t initiator;
	struct aml_replicaset_hwloc_data *data = NULL;

	data = (struct aml_replicaset_hwloc_data *)replicaset->data;

	err = aml_hwloc_local_initiator(&initiator);
	if (err != AML_SUCCESS)
		return NULL;
	while (initiator != NULL &&
	       hwloc_get_nbobjs_by_depth(aml_topology, initiator->depth) >
	               replicaset->n)
		initiator = initiator->parent;

	if (initiator == NULL)
		return NULL;
	if (hwloc_get_nbobjs_by_depth(aml_topology, initiator->depth) <
	    data->num_ptr)
		return NULL;

	return data->ptr[initiator->logical_index];
}

struct aml_replicaset_ops aml_replicaset_hwloc_ops = {
        .init = aml_replicaset_hwloc_init,
        .sync = aml_replicaset_hwloc_sync,
};
