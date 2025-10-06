/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include "config.h"

#include "aml.h"

#include "aml/higher/replicaset.h"
#include "aml/higher/replicaset/hwloc.h"
#include "aml/utils/backend/hwloc.h"

int aml_replicaset_hwloc_alloc(struct aml_replicaset **out)
{
	// Allocation + zero
	const unsigned int n_numa =
	        hwloc_get_nbobjs_by_type(aml_topology, HWLOC_OBJ_NUMANODE);
	const size_t size = sizeof(struct aml_replicaset) +
	                    sizeof(struct aml_replicaset_hwloc_data) +
	                    n_numa * sizeof(void *) +
	                    n_numa * sizeof(aml_replicaset_hwloc_memattr_t);
	unsigned char *mem = (unsigned char *)calloc(1, size);
	if (mem == NULL)
		return -AML_ENOMEM;

	struct aml_replicaset *replicaset = (struct aml_replicaset *)(mem + 0);
	struct aml_replicaset_hwloc_data *data =
	        (struct aml_replicaset_hwloc_data
	                 *)(mem + sizeof(struct aml_replicaset));

	// Set ops
	replicaset->ops = &aml_replicaset_hwloc_ops;

	// Set replica pointers array
	replicaset->replica =
	        (void **)(mem + sizeof(struct aml_replicaset) +
	                  sizeof(struct aml_replicaset_hwloc_data));

	// Set data
	replicaset->data = (struct aml_replicaset_data *)data;
	data->numas = (aml_replicaset_hwloc_memattr_t
	                       *)(mem + sizeof(struct aml_replicaset) +
	                          sizeof(struct aml_replicaset_hwloc_data) +
	                          n_numa * sizeof(void *));

	// save
	*out = replicaset;

	return AML_SUCCESS;
}

static int aml_memattr_cmp(const aml_replicaset_hwloc_memattr_t *x,
                           const aml_replicaset_hwloc_memattr_t *y)
{
	return (x->attr < y->attr ? -1 : x->attr > y->attr ? 1 : 0);
}

static int aml_memattr_cmp_inv(const aml_replicaset_hwloc_memattr_t *x,
                           const aml_replicaset_hwloc_memattr_t *y)
{
	return (x->attr < y->attr ? 1 : x->attr > y->attr ? -1 : 0);
}

int aml_replicaset_hwloc_create(struct aml_replicaset **out,
                                const size_t size,
                                const enum aml_replicaset_attr_kind_e kind)
{
	assert(kind == AML_REPLICASET_ATTR_LATENCY ||
	       kind == AML_REPLICASET_ATTR_BANDWIDTH);

	if (out == NULL)
		return -AML_FAILURE;

	// Allocate replicaset
	int err = aml_replicaset_hwloc_alloc(out);
	if (err != AML_SUCCESS)
		return err;
	(*out)->size = size;
	struct aml_replicaset_hwloc_data *data =
	        (struct aml_replicaset_hwloc_data *)(*out)->data;

	// (1) create memattr lists in the form: [(numa node x0, cpuset c0, attr
	// a0), ...]
	const unsigned int n_numa =
	        hwloc_get_nbobjs_by_type(aml_topology, HWLOC_OBJ_NUMANODE);
	for (unsigned int i = 0; i < n_numa; ++i) {
		// get the numa node
		hwloc_obj_t numa = hwloc_get_obj_by_type(aml_topology,
		                                         HWLOC_OBJ_NUMANODE, i);

		// get the attributie
		hwloc_uint64_t value;
		hwloc_memattr_id_t attr_id;
		const char *attr_name = (kind == AML_REPLICASET_ATTR_LATENCY) ?
		                                "Latency" :
		                                "Bandwidth";
		if (hwloc_memattr_get_by_name(aml_topology, attr_name,
		                              &attr_id) == 0) {
			struct hwloc_location initiator = {
			        .type = HWLOC_LOCATION_TYPE_CPUSET,
			        .location.cpuset = numa->cpuset};
			unsigned long flags = 0;
			if (hwloc_memattr_get_value(aml_topology, attr_id, numa,
			                            &initiator, flags, &value))
				value = 0;
		} else {
			value = 0;
		}

		data->numas[i].obj = numa;
		data->numas[i].attr = value;
	}

	// (2) sort the list by attr
    int (*cmpf)(const void *, const void *) = (int (*)(const void *, const void *)) (kind == AML_REPLICASET_ATTR_LATENCY ? aml_memattr_cmp : aml_memattr_cmp_inv);
	qsort(data->numas, n_numa, sizeof(aml_replicaset_hwloc_memattr_t), cmpf);

	// (3) iterate through it to find a union of numa node that covers all
	// cpus (4) replicate memory on each numa node
	hwloc_const_cpuset_t full_cpuset =
	        hwloc_topology_get_topology_cpuset(aml_topology);
	hwloc_bitmap_t cpuset = hwloc_bitmap_alloc();
	for (unsigned int i = 0; i < n_numa; ++i) {
		hwloc_bitmap_or(cpuset, cpuset, data->numas[i].obj->cpuset);
		(*out)->n += 1;
		(*out)->replica[i] = hwloc_alloc_membind(
		        aml_topology,
		        size, // size_t size
		        data->numas[i].obj->nodeset, // hwloc_nodeset_t
		        HWLOC_MEMBIND_BIND, // force binding strictly to the
		                            // node
		        HWLOC_MEMBIND_STRICT |
		                HWLOC_MEMBIND_BYNODESET // fail if can't bind
		                                        // exactly
		);
		assert((*out)->replica[i]);

		// if all cpus are represented in one replica, stop here
		if (hwloc_bitmap_isincluded(full_cpuset, cpuset)) {
			break;
		}
	}
	hwloc_bitmap_free(cpuset);

	return AML_SUCCESS;
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

// a tls for the cpuset, to avoid reallocating on every
// `aml_replicaset_hwloc_local_replica` calls
_Thread_local hwloc_bitmap_t aml_tls_cpuset = NULL;

void *aml_replicaset_hwloc_local_replica(struct aml_replicaset *replicaset)
{
	// Sanitary checks
	if (replicaset == NULL || replicaset->data == NULL)
		return NULL;

	struct aml_replicaset_hwloc_data *data =
	        (struct aml_replicaset_hwloc_data *)replicaset->data;

	// Allocate and retrieve the cpuset if not allocated already
	// If the cpuset changes for the current thread, then the replicate
	// won't... but the overhead of `hwloc_get_cpubind` is non-neglectable
	if (aml_tls_cpuset == NULL) {
		aml_tls_cpuset = hwloc_bitmap_alloc();

		// Get the cpuset of the current thread
		if (hwloc_get_cpubind(aml_topology, aml_tls_cpuset,
		                      HWLOC_CPUBIND_THREAD))
			return NULL;
	}

	// Go through the list of replicate
	for (unsigned int i = 0; i < replicaset->n; ++i) {
		// Return the first replicate that matches the current cpuset
		if (hwloc_bitmap_intersects(data->numas[i].obj->cpuset,
		                            aml_tls_cpuset))
			return replicaset->replica[i];
	}

	return NULL;
}

struct aml_replicaset_ops aml_replicaset_hwloc_ops = {
        .init = aml_replicaset_hwloc_init,
        .sync = aml_replicaset_hwloc_sync,
};
