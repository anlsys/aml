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

#include <sys/mman.h>

#include "aml.h"

#include "aml/area/hwloc.h"
#include "aml/utils/backend/hwloc.h"

#define HWLOC_BINDING_FLAGS                                                    \
	(HWLOC_MEMBIND_PROCESS | HWLOC_MEMBIND_NOCPUBIND |                     \
	 HWLOC_MEMBIND_BYNODESET | HWLOC_MEMBIND_STRICT)

#define OBJ_DIST(dist, i, j, row_stride, col_stride)                           \
	(dist)->values[((i)->logical_index + row_stride) * (dist)->nbobjs +    \
	               col_stride + (j)->logical_index]

#define IND_DIST(dist, i, j) (dist)->values[(i) * (dist)->nbobjs + (j)]

/**************************************************************************/
/* aml_area_hwloc                                                         */
/**************************************************************************/

static int aml_area_hwloc_init(struct aml_area *area,
                               hwloc_bitmap_t nodeset,
                               const hwloc_membind_policy_t policy)
{
	struct aml_area_hwloc_data *d =
	        (struct aml_area_hwloc_data *)area->data;

	// Check support
	const struct hwloc_topology_support *sup =
	        hwloc_topology_get_support(aml_topology);
	if (policy == HWLOC_MEMBIND_BIND && !sup->membind->bind_membind)
		return -AML_ENOTSUP;
	if (policy == HWLOC_MEMBIND_FIRSTTOUCH &&
	    !sup->membind->firsttouch_membind)
		return -AML_ENOTSUP;
	if (policy == HWLOC_MEMBIND_INTERLEAVE &&
	    !sup->membind->interleave_membind)
		return -AML_ENOTSUP;
	if (policy == HWLOC_MEMBIND_NEXTTOUCH &&
	    !sup->membind->nexttouch_membind)
		return -AML_ENOTSUP;

	// Check nodeset does not include unallowed nodes
	if (nodeset && !hwloc_bitmap_isincluded(nodeset, allowed_nodeset))
		return -AML_EDOM;

	// Set area nodeset and policy
	if (nodeset == NULL)
		d->nodeset = hwloc_bitmap_dup(allowed_nodeset);
	else
		d->nodeset = hwloc_bitmap_dup(nodeset);
	d->policy = policy;

	return AML_SUCCESS;
}

int aml_area_hwloc_create(struct aml_area **area,
                          hwloc_bitmap_t nodeset,
                          const hwloc_membind_policy_t policy)
{
	int err = AML_SUCCESS;
	struct aml_area *a;
	struct aml_area_hwloc_data *data;

	a = AML_INNER_MALLOC(struct aml_area, struct aml_area_hwloc_data);
	if (a == NULL)
		return -AML_ENOMEM;

	a->ops = &aml_area_hwloc_ops;
	data = AML_INNER_MALLOC_GET_FIELD(a, 2, struct aml_area,
	                                  struct aml_area_hwloc_data);
	a->data = (struct aml_area_data *)data;

	err = aml_area_hwloc_init(a, nodeset, policy);
	if (err != AML_SUCCESS) {
		free(a);
		return err;
	}
	*area = a;
	return AML_SUCCESS;
}

void aml_area_hwloc_destroy(struct aml_area **area)
{
	if (area == NULL || *area == NULL)
		return;

	struct aml_area_hwloc_data *data =
	        (struct aml_area_hwloc_data *)(*area)->data;

	hwloc_bitmap_free(data->nodeset);
	free(*area);
	*area = NULL;
}

void *aml_area_hwloc_mmap(const struct aml_area_data *data,
                          size_t size,
                          struct aml_area_mmap_options *opts)
{
	(void)opts;
	struct aml_area_hwloc_data *hwloc_data =
	        (struct aml_area_hwloc_data *)data;

	hwloc_const_bitmap_t nodeset = hwloc_data->nodeset == NULL ?
	                                       allowed_nodeset :
	                                       hwloc_data->nodeset;

	void *p = hwloc_alloc_membind(aml_topology, size, nodeset,
	                              hwloc_data->policy, HWLOC_BINDING_FLAGS);

	if (p == NULL) {
		if (errno == EINVAL)
			aml_errno = AML_EINVAL;
		if (errno == ENOSYS)
			aml_errno = AML_ENOTSUP;
		if (errno == EXDEV)
			aml_errno = AML_FAILURE;
		if (errno == ENOMEM)
			aml_errno = AML_ENOMEM;
		return NULL;
	}

	return p;
}

int aml_area_hwloc_munmap(const struct aml_area_data *data,
                          void *ptr,
                          size_t size)
{
	(void)data;
	hwloc_free(aml_topology, ptr, size);
	return AML_SUCCESS;
}

int aml_area_hwloc_check_binding(struct aml_area *area, void *ptr, size_t size)
{
	int err;
	hwloc_membind_policy_t policy;
	hwloc_bitmap_t nodeset;
	struct aml_area_hwloc_data *bind =
	        (struct aml_area_hwloc_data *)area->data;

	nodeset = hwloc_bitmap_alloc();

	if (nodeset == NULL)
		return -AML_ENOMEM;

	err = hwloc_get_area_membind(aml_topology, ptr, size, nodeset, &policy,
	                             HWLOC_BINDING_FLAGS);

	if (err < 0) {
		err = AML_EINVAL;
		goto out;
	}

	err = 1;

	if (policy != bind->policy && bind->policy != HWLOC_MEMBIND_DEFAULT &&
	    bind->policy != HWLOC_MEMBIND_MIXED)
		err = 0;

	if (!hwloc_bitmap_isequal(nodeset, bind->nodeset))
		err = 0;

out:
	hwloc_bitmap_free(nodeset);
	return err;
}

/**
 * Static declaration of an area.
 * No check is performed to enforced availability of policies and nodeset.
 * If bad values are provided area function call will return
 * an error.
 * @param[in] name: The area instance name.
 * @param[in] bitmap: The nodeset to set inside area (hwloc_bitmap_t).
 *        Can be NULL.
 * @param[in] bind_policy: The memory allocation policy
 *        (hwloc_membind_policy_t).
 **/
#define AML_AREA_HWLOC_DECL(name, bitmap, bind_policy)                         \
	struct aml_area_hwloc_data __##name##_data = {.nodeset = bitmap,       \
	                                              .policy = bind_policy};  \
	struct aml_area name = {                                               \
	        .ops = &aml_area_hwloc_ops,                                    \
	        .data = (struct aml_area_data *)&__##name##_data}

struct aml_area_ops aml_area_hwloc_ops = {
        .mmap = aml_area_hwloc_mmap,
        .munmap = aml_area_hwloc_munmap,
        .fprintf = NULL,
};

AML_AREA_HWLOC_DECL(aml_area_hwloc_default, NULL, HWLOC_MEMBIND_DEFAULT);

AML_AREA_HWLOC_DECL(aml_area_hwloc_interleave, NULL, HWLOC_MEMBIND_INTERLEAVE);

AML_AREA_HWLOC_DECL(aml_area_hwloc_firsttouch, NULL, HWLOC_MEMBIND_FIRSTTOUCH);

/**************************************************************************/
/* aml_hwloc_preferred_policy                                             */
/**************************************************************************/

static int aml_area_hwloc_preferred_alloc(struct aml_area **area)
{
	if (area == NULL)
		return -AML_EINVAL;
	int num_nodes =
	        hwloc_get_nbobjs_by_type(aml_topology, HWLOC_OBJ_NUMANODE);
	if (num_nodes <= 0)
		return -AML_ENOTSUP;

	struct aml_area *a =
	        AML_INNER_MALLOC_ARRAY(num_nodes, hwloc_obj_t, struct aml_area,
	                               struct aml_area_hwloc_preferred_data);

	if (a == NULL)
		return -AML_ENOMEM;

	struct aml_area_hwloc_preferred_data *policy =
	        AML_INNER_MALLOC_GET_FIELD(
	                a, 2, struct aml_area,
	                struct aml_area_hwloc_preferred_data);

	policy->numanodes = AML_INNER_MALLOC_GET_ARRAY(
	        a, hwloc_obj_t, struct aml_area,
	        struct aml_area_hwloc_preferred_data);
	policy->num_nodes = (unsigned)num_nodes;

	for (int i = 0; i < num_nodes; i++)
		policy->numanodes[i] = NULL;

	a->data = (struct aml_area_data *)policy;
	a->ops = &aml_area_hwloc_preferred_ops;
	*area = a;
	return AML_SUCCESS;
}

int aml_area_hwloc_preferred_destroy(struct aml_area **area)
{
	if (area != NULL && *area != NULL) {
		free(*area);
		*area = NULL;
	}
	return AML_SUCCESS;
}

void *aml_area_hwloc_preferred_mmap(const struct aml_area_data *area_data,
                                    size_t size,
                                    struct aml_area_mmap_options *opts)

{

	struct aml_area_hwloc_options *options =
	        (struct aml_area_hwloc_options *)opts;
	struct aml_area_hwloc_preferred_data *data =
	        (struct aml_area_hwloc_preferred_data *)area_data;

	// Allocate data
	void *ptr = mmap(NULL, size, PROT_READ | PROT_WRITE,
	                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

	if (ptr == NULL || ptr == MAP_FAILED) {
		aml_errno = AML_ENOMEM;
		return NULL;
	}

	int err;
	intptr_t offset = 0;
	hwloc_obj_t *numanode = data->numanodes;
	size_t block_size;

	// Map block per block. Stop if we reached beyond last numanode.
	while ((size_t)offset < size &&
	       numanode < (data->numanodes + data->num_nodes)) {
		// Compute block size based on numanode page size and options.
		block_size = (*numanode)->attr->numanode.page_types->size;
		if (options != NULL)
			block_size = options->min_size -
			             (options->min_size % block_size) +
			             block_size;

		// Apply membind
		err = hwloc_set_area_membind(
		        aml_topology, (void *)((intptr_t)ptr + offset),
		        block_size, (*numanode)->nodeset, HWLOC_MEMBIND_BIND,
		        HWLOC_MEMBIND_STRICT | HWLOC_MEMBIND_BYNODESET);

		// If not possible then try next numanode.
		if (err < 0) {
			numanode++;
		} else {
			offset += block_size;
		}
	}

	return ptr;
}

int aml_area_hwloc_preferred_munmap(const struct aml_area_data *data,
                                    void *ptr,
                                    size_t size)
{
	(void)data;
	munmap(ptr, size);
	return AML_SUCCESS;
}

/**
 *  SPR memories bandwidth is not exposed or accessible by hwloc, therefore it
 *  fails when checking distances.  This ugly fix consists in getting latency
 *  distances instead, and swapping ddr/hbm from their hard-codded index
 */
# define INTEL_SPR_UGLY_FIX 1

int aml_area_hwloc_preferred_create(struct aml_area **area,
                                    hwloc_obj_t initiator,
                                    enum hwloc_distances_kind_e kind)
{
	// Check input
	if (area == NULL)
		return -AML_EINVAL;
	if (initiator == NULL)
		return -AML_EINVAL;

    # if INTEL_SPR_UGLY_FIX
    int using_bandwidth_distance = (kind & HWLOC_DISTANCES_KIND_VALUE_BANDWIDTH);
    if (using_bandwidth_distance)
    {
        assert(!(kind & HWLOC_DISTANCES_KIND_VALUE_LATENCY));
        kind = (kind & ~HWLOC_DISTANCES_KIND_VALUE_BANDWIDTH) | HWLOC_DISTANCES_KIND_VALUE_LATENCY;
        using_bandwidth_distance = 1;
    }
    # endif /* INTEL_SPR_UGLY_FIX */

	// The number of nodes in this system.
	const unsigned num_nodes =
	        hwloc_get_nbobjs_by_type(aml_topology, HWLOC_OBJ_NUMANODE);
	// The number of initiator
	const unsigned num_initiator =
	        hwloc_get_nbobjs_by_type(aml_topology, initiator->type);
	// output area
	struct aml_area *ar = NULL;
	// area numanodes
	struct aml_area_hwloc_preferred_data *data;
	// Distances
	struct hwloc_distances_s *dist;
	// array of distances to sort
	struct aml_hwloc_distance distances[num_nodes];
	// distances from/to initiator, to/from target.
	hwloc_uint64_t itot = 0, ttoi = 0;

    // Allocate structures
	aml_area_hwloc_preferred_alloc(&ar);
	if (ar == NULL)
		return -AML_ENOMEM;
	data = (struct aml_area_hwloc_preferred_data *)ar->data;

	// Collect distances
	if (aml_hwloc_get_NUMA_distance(initiator->type, kind, &dist) !=
	    AML_SUCCESS)
		return -AML_ENOMEM;

	// For each numanode compute distance to initiator
	for (unsigned i = 0; i < num_nodes; i++) {
		hwloc_obj_t target = hwloc_get_obj_by_type(
		        aml_topology, HWLOC_OBJ_NUMANODE, i);
		if (initiator->type == HWLOC_OBJ_NUMANODE) {
			itot = OBJ_DIST(dist, initiator, target, 0, 0);
			ttoi = OBJ_DIST(dist, target, initiator, 0, 0);
		} else {
			itot = OBJ_DIST(dist, initiator, target, 0,
			                num_initiator);
			ttoi = OBJ_DIST(dist, target, initiator, num_initiator,
			                0);
		}

		// Store average distance (back and forth)
		distances[i].distance = (itot + ttoi) / 2;
		distances[i].index = i;
	}

	// Sort distances
	qsort(distances, num_nodes, sizeof(*distances), aml_hwloc_distance_lt);
	// Store sorted nodes in area data.
	for (unsigned i = 0; i < num_nodes; i++) {
		hwloc_obj_t node = hwloc_get_obj_by_type(
		        aml_topology, HWLOC_OBJ_NUMANODE, distances[i].index);
		data->numanodes[i] = node;
	}

    # if INTEL_SPR_UGLY_FIX
    if (using_bandwidth_distance)
    {
        hwloc_obj_t o0 = data->numanodes[0];    // my    ddr
        hwloc_obj_t o1 = data->numanodes[1];    // my    hbm
        hwloc_obj_t o2 = data->numanodes[2];    // other ddr
        hwloc_obj_t o3 = data->numanodes[3];    // other hbm
        data->numanodes[0] = o1;
        data->numanodes[1] = o0;
        data->numanodes[2] = o3;
        data->numanodes[3] = o2;
    }
    # endif /* INTEL_SPR_UGLY_FIX */

	// Cleanup
	free(dist);
	// Success !
	*area = ar;
	return AML_SUCCESS;
}

int aml_area_hwloc_preferred_local_create(struct aml_area **area,
                                          enum hwloc_distances_kind_e kind)
{
	int err;
	hwloc_obj_t initiator;

	err = aml_hwloc_local_initiator(&initiator);
	if (err != AML_SUCCESS)
		return err;

	/** Build area with found initiator **/
	return aml_area_hwloc_preferred_create(area, initiator, kind);
}

struct aml_area_ops aml_area_hwloc_preferred_ops = {
        .mmap = aml_area_hwloc_preferred_mmap,
        .munmap = aml_area_hwloc_preferred_munmap,
        .fprintf = NULL,
};
