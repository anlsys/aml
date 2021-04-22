/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#define _GNU_SOURCE
#include <aml.h>
#include <numa.h>
#include <numaif.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

// Function to get last NUMA node id on which data is allocated.
static int
get_node(void *data)
{
	long err;
	int policy;
	unsigned long maxnode = sizeof(unsigned long) * 8;
	unsigned long nmask   = 0;
	int node	      = -1;

	err = get_mempolicy(&policy, &nmask, maxnode, data, MPOL_F_ADDR);
	if (err == -1) {
		perror("get_mempolicy");
		exit(1);
	}

	while (nmask != 0) {
		node++;
		nmask = nmask >> 1;
	}

	return node;
}

// Check if data of size `size` is interleaved on all nodes,
// by chunk of size `page_size`.
static int
is_interleaved(void *data, const size_t size, const size_t page_size)
{
	intptr_t start;
	int node, next, num_nodes = 0;

	start = ((intptr_t)data) << page_size >> page_size;

	node = get_node((void *)start);
	// more than one node in policy.
	if (node < 0)
		return 0;

	for (intptr_t page = start + page_size; (size_t)(page - start) < size;
	     page += page_size) {
		next = get_node((void *)page);

		// more than one node in page policy.
		if (next < 0)
			return 0;
		// not round-robin
		if (next != (node + 1) && next != 0)
			return 0;
		// cycling on different number of nodes
		if (num_nodes != 0 && next >= num_nodes)
			return 0;
		// cycling on different number of nodes
		if (num_nodes != 0 && next == 0 && num_nodes != node)
			return 0;
		// set num_nodes
		if (num_nodes == 0 && next == 0)
			num_nodes = node;
		node = next;
	}

	return 1;
}

// Custom area attributes.
struct area_data {
	unsigned long nid;  // Current node id;
	unsigned long nmax; // highest node number on this system.
	int page_size;      // Size of a page;
};

// Custom area mmap implementation
void *
custom_mmap(const struct aml_area_data *data,
	    size_t size,
	    struct aml_area_mmap_options *opts)
{
	(void)opts;
	intptr_t start;
	struct area_data *area = (struct area_data *)data;
	void *ret	      = mmap(NULL,
			      size,
			      PROT_READ | PROT_WRITE,
			      MAP_PRIVATE | MAP_ANONYMOUS,
			      0,
			      0);

	if (ret == NULL)
		return NULL;

	start = (intptr_t)ret >> area->page_size << area->page_size;

	for (intptr_t page = start; page < (start + (intptr_t)size);
	     page += area->page_size) {
		if (mbind((void *)page,
			  area->page_size,
			  MPOL_BIND,
			  &area->nid,
			  8 * sizeof(unsigned long),
			  MPOL_MF_MOVE) == -1) {
			perror("mbind");
			munmap(ret, size);
			return NULL;
		}
		area->nid = area->nid << 1;
		if (area->nid >= (1UL << area->nmax))
			area->nid = 1;
	}

	return ret;
}

// Custom area munmap implementation
int
custom_munmap(const struct aml_area_data *data, void *ptr, size_t size)
{
	(void)data;
	munmap(ptr, size);
	return AML_SUCCESS;
}

// Custom area constructor
struct aml_area *
custom_area_create()
{
	struct area_data *data;
	struct aml_area_ops *ops;
	struct aml_area *ret = AML_INNER_MALLOC(
	    struct aml_area, struct aml_area_ops, struct area_data);
	if (ret == NULL)
		return NULL;

	ret->ops = AML_INNER_MALLOC_GET_FIELD(
	    ret, 2, struct aml_area, struct aml_area_ops, struct area_data);
	ops	 = ret->ops;
	ops->mmap   = custom_mmap;
	ops->munmap = custom_munmap;

	ret->data = AML_INNER_MALLOC_GET_FIELD(
	    ret, 3, struct aml_area, struct aml_area_ops, struct area_data);
	data		= (struct area_data *)ret->data;
	data->nid       = 1;
	data->nmax      = numa_max_node() == 0 ? 1 : numa_max_node();
	data->page_size = 2 * sysconf(_SC_PAGESIZE); // 2 pages allocation

	return ret;
}

// Test that custom area will interleave pages as expected.
void
test_custom_area(const size_t size)
{
	void *buf;
	struct aml_area *interleave_area = custom_area_create();

	// Map buffer in area.
	buf = aml_area_mmap(interleave_area, size, NULL);
	if (buf == NULL) {
		aml_perror("aml_area_linux");
		exit(1);
	}
	// Check it is indeed interleaved
	if (!is_interleaved(buf, size, 2 * sysconf(_SC_PAGESIZE)))
		exit(1);
	printf("Custom area worked and is interleaved.\n");

	// Cleanup
	aml_area_munmap(interleave_area, buf, size);
	free(interleave_area);
}

int
main(void)
{
	const size_t size = (2 << 16); // 16 pages

	test_custom_area(size);
	return 0;
}
