/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include <sys/mman.h>

#include "aml.h"

#include "aml/area/hwloc.h"

#define HWLOC_BINDING_FLAGS                                                    \
	(HWLOC_MEMBIND_PROCESS | HWLOC_MEMBIND_NOCPUBIND |                     \
	 HWLOC_MEMBIND_BYNODESET | HWLOC_MEMBIND_STRICT)

/**
 * Topology used by AML.
 * Topology is initalized when calling aml_library_init().
 **/
extern hwloc_topology_t aml_topology;
extern hwloc_const_bitmap_t allowed_nodeset;
typedef const struct hwloc_obj *hwloc_const_obj_t;

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
	void *ptr =
	        mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE, -1, 0);

	if (ptr == NULL) {
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

// Struct for sorting (distance) and reordering (index)
struct aml_area_hwloc_distance {
	unsigned index;
	hwloc_uint64_t distance;
};

/**
 * Get distance in hops between two objects of the same topology.
 * @param a: An hwloc object with positive depth or NUMANODE.
 * @param b: An hwloc object with positive depth or NUMANODE.
 * @return the distance in hops between two objects.
 * @return -1 if distance cannot be computed.
 **/
static int aml_area_hwloc_hwloc_lt(const void *a_ptr, const void *b_ptr)
{
	struct aml_area_hwloc_distance *a =
	        (struct aml_area_hwloc_distance *)a_ptr;
	struct aml_area_hwloc_distance *b =
	        (struct aml_area_hwloc_distance *)b_ptr;
	return a->distance < b->distance;
}

/**
 * Allocate elements separately because when they are added to the
 * topology, hwloc free them separately.
 **/
static int aml_hwloc_distances_alloc(const hwloc_obj_type_t t0,
                                     const hwloc_obj_type_t t1,
                                     struct hwloc_distances_s **out,
                                     unsigned *nt0,
                                     unsigned *nt1)
{
	unsigned n;

	*nt0 = hwloc_get_nbobjs_by_type(aml_topology, t0);
	if (t0 == t1) {
		*nt1 = *nt0;
		n = *nt0;
	} else {
		*nt1 = hwloc_get_nbobjs_by_type(aml_topology, t1);
		n = *nt0 + *nt1;
	}

	*out = malloc(sizeof(**out));
	if (*out == NULL)
		return -AML_ENOMEM;

	(*out)->objs = malloc(n * sizeof(hwloc_obj_t));
	if ((*out)->objs == NULL)
		goto err_with_distances;

	(*out)->values = malloc(n * n * sizeof(*((*out)->values)));
	if ((*out)->values == NULL)
		goto err_with_objs;

	(*out)->nbobjs = n;

	for (unsigned it0 = 0; it0 < *nt0; it0++)
		(*out)->objs[it0] =
		        hwloc_get_obj_by_type(aml_topology, t0, it0);

	if (t0 != t1) {
		for (unsigned it1 = 0; it1 < *nt1; it1++)
			(*out)->objs[*nt0 + it1] =
			        hwloc_get_obj_by_type(aml_topology, t1, it1);
	}
	return AML_SUCCESS;

err_with_objs:
	free((*out)->objs);
err_with_distances:
	free(*out);
	return -AML_ENOMEM;
}

#define OBJ_DIST(dist, i, j, row_stride, col_stride)                           \
	(dist)->values[((i)->logical_index + row_stride) * (dist)->nbobjs +    \
	               col_stride + (j)->logical_index]

#define IND_DIST(dist, i, j) (dist)->values[(i) * (dist)->nbobjs + (j)]

static void aml_hwloc_distances_free(struct hwloc_distances_s *dist)
{
	free(dist->objs);
	free(dist->values);
	free(dist);
}

/**
 * Get distance matrix in hops between two types of objects.
 * Distance matrix comes in the format used in <hwloc/distance.h>.
 * [ objs_a , objs_b ] * [ objs_a , objs_b ] such that element
 * matrix [ i, j ] =  i * (nbobj_a + nbobj_b)  + j
 * @param[in] ta: An hwloc object type with positive depth or NUMANODE.
 * @param[in] tb: An hwloc object type with positive depth or NUMANODE.
 * @param[out] distances: A pointer to a distance struct that will be
 * populated with hop distances and that can be freed with
 * hwloc_distances_release.
 * @return 0 on success.
 * @return -1 on failure.
 **/
static int aml_hwloc_distance_hop(hwloc_const_obj_t a, hwloc_const_obj_t b)
{
	int dist = 0;

	if (a == b)
		return 0;
	if (a->type == HWLOC_OBJ_NUMANODE) {
		a = a->parent;
		dist += 1;
	}
	if (b->type == HWLOC_OBJ_NUMANODE) {
		b = b->parent;
		dist += 1;
	}
	if (a->depth < 0)
		return -1;
	if (b->depth < 0)
		return -1;

	// Go up from a until b cpuset is included into a
	while (a != NULL && !hwloc_bitmap_isincluded(b->cpuset, a->cpuset)) {
		dist++;
		a = a->parent;
	}

	if (a == NULL)
		return -1;

	// Go up from b until a cpuset is equal to a cpuset
	while (b != NULL && !hwloc_bitmap_isequal(b->cpuset, a->cpuset)) {
		dist++;
		b = b->parent;
	}

	if (b == NULL)
		return -1;
	return dist;
}

/**
 * Create a distance matrix of topology hops between to object types.
 **/
int aml_hwloc_distance_hop_matrix(const hwloc_obj_type_t ta,
                                  const hwloc_obj_type_t tb,
                                  struct hwloc_distances_s **s)
{
	hwloc_uint64_t d;
	unsigned na, nb;

	if (aml_hwloc_distances_alloc(ta, tb, s, &na, &nb) != AML_SUCCESS)
		return -AML_ENOMEM;

	(*s)->kind = HWLOC_DISTANCES_KIND_FROM_USER |
	             HWLOC_DISTANCES_KIND_MEANS_LATENCY;

	// Store distances of same type a
	for (unsigned i = 0; i < na; i++)
		for (unsigned j = i; j < na; j++) {
			d = aml_hwloc_distance_hop(
			        hwloc_get_obj_by_type(aml_topology, ta, i),
			        hwloc_get_obj_by_type(aml_topology, ta, j));
			IND_DIST(*s, i, j) = d;
			IND_DIST(*s, j, i) = d;
		}

	// If both types are equal, then we stored everything
	if (ta == tb)
		return 0;
	else
		(*s)->kind |= HWLOC_DISTANCES_KIND_HETEROGENEOUS_TYPES;

	// Store distances of same type b
	for (unsigned i = 0; i < nb; i++)
		for (unsigned j = i; j < nb; j++) {
			d = aml_hwloc_distance_hop(
			        hwloc_get_obj_by_type(aml_topology, tb, i),
			        hwloc_get_obj_by_type(aml_topology, tb, j));
			IND_DIST(*s, na + i, na + j) = d;
			IND_DIST(*s, na + j, na + i) = d;
		}

	// Store distances  ab, ba
	for (unsigned i = 0; i < na; i++)
		for (unsigned j = 0; j < nb; j++) {
			d = aml_hwloc_distance_hop(
			        hwloc_get_obj_by_type(aml_topology, ta, i),
			        hwloc_get_obj_by_type(aml_topology, tb, j));
			IND_DIST(*s, na + j, i) = d;
			IND_DIST(*s, i, na + j) = d;
		}

	return 0;
}

/**
 * Take a distance matrix with a single type T0 of object and return
 * a derived distance between two objects. Distances sum if
 * T0 have several children of input object types.
 * This allow to compute a distance between any 2 arbitrary objects
 * any distance matrix exists.
 **/
static int aml_hwloc_distance_match(struct hwloc_distances_s *dist,
                                    const hwloc_obj_t obj0,
                                    const hwloc_obj_t obj1,
                                    hwloc_uint64_t *d0,
                                    hwloc_uint64_t *d1)
{
	hwloc_obj_t l0, l1;
	const unsigned depth = dist->objs[0]->depth;

	*d0 = 0;
	*d1 = 0;

	l0 = hwloc_get_next_obj_covering_cpuset_by_depth(
	        aml_topology, obj0->cpuset, depth, NULL);
	do {
		l1 = hwloc_get_next_obj_covering_cpuset_by_depth(
		        aml_topology, obj1->cpuset, depth, NULL);
		do {
			*d0 += OBJ_DIST(dist, l0, l1, 0, 0);
			*d1 += OBJ_DIST(dist, l1, l0, 0, 0);
			l1 = hwloc_get_next_obj_covering_cpuset_by_depth(
			        aml_topology, obj1->cpuset, depth, l1);
		} while (l1 != NULL);
		l0 = hwloc_get_next_obj_covering_cpuset_by_depth(
		        aml_topology, obj0->cpuset, depth, l0);
	} while (l0 != NULL);
	return 0;
}

/**
 * Take a distance matrix and convert it to another matrix with different
 * object types.
 **/
static int aml_hwloc_distances_reshape(struct hwloc_distances_s *dist,
                                       struct hwloc_distances_s **out,
                                       const hwloc_obj_type_t t0,
                                       const hwloc_obj_type_t t1)
{
	if (dist->kind & HWLOC_DISTANCES_KIND_HETEROGENEOUS_TYPES)
		return -AML_EINVAL;

	unsigned nt0, nt1;
	const unsigned depth0 = hwloc_get_type_depth(aml_topology, t0);
	const unsigned depth1 = hwloc_get_type_depth(aml_topology, t1);
	hwloc_obj_t obj0 = NULL, obj1 = NULL;
	hwloc_uint64_t d0, d1;

	if (aml_hwloc_distances_alloc(t0, t1, out, &nt0, &nt1) != AML_SUCCESS)
		return -AML_ENOMEM;

	// Set kind.
	(*out)->kind = HWLOC_DISTANCES_KIND_FROM_USER;
	if (dist->kind & HWLOC_DISTANCES_KIND_MEANS_LATENCY)
		(*out)->kind |= HWLOC_DISTANCES_KIND_MEANS_LATENCY;
	if (dist->kind & HWLOC_DISTANCES_KIND_MEANS_BANDWIDTH)
		(*out)->kind |= HWLOC_DISTANCES_KIND_MEANS_BANDWIDTH;

	// Set distances t0 <-> t0.
	obj0 = hwloc_get_next_obj_by_depth(aml_topology, depth0, NULL);
	while (obj0 != NULL) {
		OBJ_DIST(*out, obj0, obj0, 0, 0) = 0;
		obj1 = hwloc_get_next_obj_by_depth(aml_topology, depth0, obj0);
		while (obj1 != NULL) {
			if (aml_hwloc_distance_match(dist, obj0, obj1, &d0,
			                             &d1) != 0)
				goto err_with_out;
			OBJ_DIST(*out, obj0, obj1, 0, 0) = d0;
			OBJ_DIST(*out, obj1, obj0, 0, 0) = d1;
			obj1 = hwloc_get_next_obj_by_depth(aml_topology, depth0,
			                                   obj1);
		}
		obj0 = hwloc_get_next_obj_by_depth(aml_topology, depth0, obj0);
	}

	if (t0 == t1)
		return AML_SUCCESS;

	(*out)->kind |= HWLOC_DISTANCES_KIND_HETEROGENEOUS_TYPES;

	// Set distances t0 <-> t1.
	obj0 = hwloc_get_next_obj_by_depth(aml_topology, depth0, NULL);
	while (obj0 != NULL) {
		obj1 = hwloc_get_next_obj_by_depth(aml_topology, depth1, NULL);
		while (obj1 != NULL) {
			if (aml_hwloc_distance_match(dist, obj0, obj1, &d0,
			                             &d1) != 0)
				goto err_with_out;
			OBJ_DIST(*out, obj0, obj1, 0, nt0) = d0;
			OBJ_DIST(*out, obj1, obj0, nt0, 0) = d1;
			obj1 = hwloc_get_next_obj_by_depth(aml_topology, depth0,
			                                   obj1);
		}
		obj0 = hwloc_get_next_obj_by_depth(aml_topology, depth0, obj0);
	}

	// Set distances t1 <-> t1.
	obj0 = hwloc_get_next_obj_by_depth(aml_topology, depth1, NULL);
	while (obj0 != NULL) {
		OBJ_DIST(*out, obj0, obj0, nt0, nt0) = 0;
		obj1 = hwloc_get_next_obj_by_depth(aml_topology, depth1, obj0);
		while (obj1 != NULL) {
			if (aml_hwloc_distance_match(dist, obj0, obj1, &d0,
			                             &d1) != 0)
				goto err_with_out;
			OBJ_DIST(*out, obj0, obj1, nt0, nt0) = d0;
			OBJ_DIST(*out, obj1, obj0, nt0, nt0) = d1;
			obj1 = hwloc_get_next_obj_by_depth(aml_topology, depth1,
			                                   obj1);
		}
		obj0 = hwloc_get_next_obj_by_depth(aml_topology, depth1, obj0);
	}

	return AML_SUCCESS;

err_with_out:
	aml_hwloc_distances_free(*out);
	return -AML_FAILURE;
}

/**
 * Get a distance matrix between an initiator type and NUMANODEs
 * This method never fails.
 * If no distance matrix exist, one is created with hops as distance.
 * If no matrix with initiator type or NUMANODE type is found, any
 * available matrix is used. The obtained matrix is reshaped to fit
 * (initiator numa) x (initiator numa) matrix.
 **/
int aml_hwloc_get_NUMA_distance(const hwloc_obj_type_t type,
                                enum hwloc_distances_kind_e kind,
                                struct hwloc_distances_s **out)
{
	// number of hwloc matrices to return in handle
	unsigned int nr = 32;
	// hwloc distance matrix
	struct hwloc_distances_s *handle[nr], *dist = NULL;

	// Collect distances. If fail, fallback on hop distances.
	if (hwloc_distances_get(aml_topology, &nr, handle, kind, 0) != 0 ||
	    nr == 0) {
		if (aml_hwloc_distance_hop_matrix(type, HWLOC_OBJ_NUMANODE,
		                                  out) == -1)
			return -1;
		return 0;
	}

	for (unsigned i = 0; i < nr; i++) {
		// If we found a matrix with same type as initiator type
		// then we pick this one.
		if (handle[i]->objs[0]->type == type) {
			dist = handle[i];
			break;
		}
		// We pick any distance
		if (dist == NULL)
			dist = handle[i];
		// If we find one that is a NUMANODE distance, we chose this one
		// over a default choice.
		if (handle[i]->objs[0]->type == HWLOC_OBJ_NUMANODE)
			dist = handle[i];
		// If we find a distance that is finer grain than default,
		// then we chose this one.
		if (dist->objs[0]->type != HWLOC_OBJ_NUMANODE &&
		    dist->objs[0]->depth < handle[i]->objs[0]->depth)
			dist = handle[i];
	}

	// If we were not able to find any matrix, we craft one.
	if (dist == NULL) {
		if (aml_hwloc_distance_hop_matrix(type, HWLOC_OBJ_NUMANODE,
		                                  out) != AML_SUCCESS)
			return -AML_ENOMEM;
		return AML_SUCCESS;
	}

	// We reshape whatever matrix we got to be a distance to NUMANODEs
	// matrix.
	if (aml_hwloc_distances_reshape(dist, out, type, HWLOC_OBJ_NUMANODE) !=
	    AML_SUCCESS)
		return -AML_ENOMEM;
	return AML_SUCCESS;
}

int aml_area_hwloc_preferred_create(struct aml_area **area,
                                    hwloc_obj_t initiator,
                                    enum hwloc_distances_kind_e kind)
{
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
	struct aml_area_hwloc_distance distances[num_nodes];
	// distances from/to initiator, to/from target.
	hwloc_uint64_t itot = 0, ttoi = 0;

	// Check input
	if (area == NULL)
		return -AML_EINVAL;
	if (initiator == NULL)
		return -AML_EINVAL;

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
	qsort(distances, num_nodes, sizeof(*distances),
	      aml_area_hwloc_hwloc_lt);
	// Store sorted nodes in area data.
	for (unsigned i = 0; i < num_nodes; i++) {
		hwloc_obj_t node = hwloc_get_obj_by_type(
		        aml_topology, HWLOC_OBJ_NUMANODE, distances[i].index);
		data->numanodes[i] = node;
	}

	// Cleanup
	aml_hwloc_distances_free(dist);

	// Success !
	*area = ar;
	return AML_SUCCESS;
}

int aml_area_hwloc_preferred_local_create(struct aml_area **area,
                                          enum hwloc_distances_kind_e kind)
{
	int err;
	hwloc_cpuset_t cpuset;
	hwloc_obj_t initiator;

	/** Collect cpuset binding of current thread **/
	cpuset = hwloc_bitmap_alloc();
	if (cpuset == NULL)
		return -AML_ENOMEM;
	err = hwloc_get_cpubind(aml_topology, cpuset, HWLOC_CPUBIND_THREAD);

	/** If thread is not bound we raise an error **/
	if (err != 0 || hwloc_bitmap_iszero(cpuset)) {
		err = -AML_FAILURE;
		goto err_with_cpuset;
	}

	/** Match cpuset with a location on machine **/
	err = hwloc_get_largest_objs_inside_cpuset(aml_topology, cpuset,
	                                           &initiator, 1);
	if (err == -1) {
		err = -AML_FAILURE;
		goto err_with_cpuset;
	}
	hwloc_bitmap_free(cpuset);

	/** Build area with found initiator **/
	return aml_area_hwloc_preferred_create(area, initiator, kind);

err_with_cpuset:
	hwloc_bitmap_free(cpuset);
	return err;
}

struct aml_area_ops aml_area_hwloc_preferred_ops = {
        .mmap = aml_area_hwloc_preferred_mmap,
        .munmap = aml_area_hwloc_preferred_munmap,
        .fprintf = NULL,
};
