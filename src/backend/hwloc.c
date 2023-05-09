/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include <hwloc.h>

#include "aml.h"

#include "aml/utils/backend/hwloc.h"

hwloc_topology_t aml_topology;
hwloc_const_bitmap_t allowed_nodeset;

#define OBJ_DIST(dist, i, j, row_offset, col_offset)                           \
	(dist)->values[((i)->logical_index + row_offset) * (dist)->nbobjs +    \
	               col_offset + (j)->logical_index]

#define IND_DIST(dist, i, j) (dist)->values[(i) * (dist)->nbobjs + (j)]

typedef const struct hwloc_obj *hwloc_const_obj_t;

int aml_backend_hwloc_init(void)
{
	char *topology_input = getenv("AML_TOPOLOGY");

	if (hwloc_topology_init(&aml_topology) == -1)
		return -AML_FAILURE;

	if (hwloc_topology_set_flags(
	            aml_topology,
	            HWLOC_TOPOLOGY_FLAG_THISSYSTEM_ALLOWED_RESOURCES |
	                    HWLOC_TOPOLOGY_FLAG_IS_THISSYSTEM) == -1)
		return -AML_FAILURE;

	if (topology_input != NULL &&
	    hwloc_topology_set_xml(aml_topology, topology_input) == -1)
		return -AML_FAILURE;

	if (hwloc_topology_load(aml_topology) == -1)
		return -AML_FAILURE;

	allowed_nodeset = hwloc_topology_get_allowed_nodeset(aml_topology);
	return AML_SUCCESS;
}

int aml_backend_hwloc_finalize(void)
{
	hwloc_topology_destroy(aml_topology);
	return AML_SUCCESS;
}

int aml_hwloc_local_initiator(hwloc_obj_t *out)
{
	int err;
	hwloc_cpuset_t cpuset;

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
	err = hwloc_get_largest_objs_inside_cpuset(aml_topology, cpuset, out,
	                                           1);
	if (err == -1) {
		err = -AML_FAILURE;
		goto err_with_cpuset;
	}

	hwloc_bitmap_free(cpuset);
	return AML_SUCCESS;

err_with_cpuset:
	hwloc_bitmap_free(cpuset);
	return err;
}

/**
 * Get distance in hops between two objects of the same topology.
 * @param a: An hwloc object with positive depth or NUMANODE.
 * @param b: An hwloc object with positive depth or NUMANODE.
 * @return the distance in hops between two objects.
 * @return -1 if distance cannot be computed.
 **/
int aml_hwloc_distance_lt(const void *a_ptr, const void *b_ptr)
{
	struct aml_hwloc_distance *a = (struct aml_hwloc_distance *)a_ptr;
	struct aml_hwloc_distance *b = (struct aml_hwloc_distance *)b_ptr;
	if (a->distance < b->distance)
		return -1;
	else if (a->distance > b->distance)
		return 1;
	else
		return 0;
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

	*out = AML_INNER_MALLOC_EXTRA(n, hwloc_obj_t,
	                              n * n * sizeof(*((*out)->values)),
	                              struct hwloc_distances_s);
	if (*out == NULL)
		return -AML_ENOMEM;
	(*out)->objs = AML_INNER_MALLOC_GET_ARRAY(*out, hwloc_obj_t,
	                                          struct hwloc_distances_s);
	(*out)->values = AML_INNER_MALLOC_GET_EXTRA(*out, n, hwloc_obj_t,
	                                            struct hwloc_distances_s);
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
			obj1 = hwloc_get_next_obj_by_depth(aml_topology, depth1,
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
	free(*out);
	return -AML_FAILURE;
}

int aml_hwloc_get_NUMA_distance(const hwloc_obj_type_t type,
                                enum hwloc_distances_kind_e kind,
                                struct hwloc_distances_s **out)
{
	int err = AML_SUCCESS;
	// number of hwloc matrices to return in handle
	unsigned int i, nr = 32;
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

	for (i = 0; i < nr; i++) {
		// We pick any distance
		if (dist == NULL)
			dist = handle[i];

		// If we found a matrix with same type as initiator type
		// then we pick this one.
		else if (handle[i]->objs[0]->type == type) {
			dist = handle[i];
			break;
		}

		// If we find one that is a NUMANODE distance, we chose this one
		// over a default choice.
		else if (handle[i]->objs[0]->type == HWLOC_OBJ_NUMANODE)
			dist = handle[i];

		// If we find a distance that is finer grain than default,
		// then we chose this one.
		else if (dist->objs[0]->type != HWLOC_OBJ_NUMANODE &&
		         dist->objs[0]->depth < handle[i]->objs[0]->depth)
			dist = handle[i];
	}

	// If we were not able to find any matrix, we craft one.
	if (dist == NULL) {
		if (aml_hwloc_distance_hop_matrix(type, HWLOC_OBJ_NUMANODE,
		                                  out) != AML_SUCCESS)
			err = -AML_ENOMEM;
		goto out;
	}

	// We reshape whatever matrix we got to be a distance to NUMANODEs
	// matrix.
	if (aml_hwloc_distances_reshape(dist, out, type, HWLOC_OBJ_NUMANODE) !=
	    AML_SUCCESS)
		err = -AML_ENOMEM;

out:
	for (i = 0; i < nr; i++)
		hwloc_distances_release(aml_topology, handle[i]);
	return err;
}
