/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#define _GNU_SOURCE
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>		// sysconf()
#include <sys/mman.h>		// mmap()
#include <numa.h>		// #define NUMA_NUM_NODES
#include <numaif.h>		// get_mempolicy()

/**
 * Check whether linux inerleave binding policy
 * is set on this range of address.
 **/
int is_interleaved_policy(void *data)
{
	long err;
	int policy;
	unsigned long maxnode = sizeof(unsigned long) * 8;
	unsigned long nmask;

	err = get_mempolicy(&policy, &nmask, maxnode, data, MPOL_F_ADDR);
	if (err == -1) {
		perror("get_mempolicy");
		exit(1);
	}

	return policy == MPOL_INTERLEAVE;
}

/**
 * Returns the numa node on which data is bound
 * or -1 if there is more than one node involved in binding.
 **/
int get_node(void *data)
{
	long err;
	int policy;
	unsigned long maxnode = sizeof(unsigned long) * 8;
	unsigned long nmask = 0;
	int node = -1;

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

/**
 * Walk all pages and check if they are bound in a round-robin
 * fashion.
 **/
int is_interleaved_bind(void *data, const size_t size)
{
	int page_size;
	intptr_t start;
	int node, next, num_nodes = 0;

	page_size = sysconf(_SC_PAGESIZE);
	start = ((intptr_t) data) << page_size >> page_size;

	node = get_node((void *)start);
	// more than one node in policy.
	if (node < 0)
		return 0;

	for (intptr_t page = start + page_size;
	     (size_t) (page - start) < size; page += page_size) {
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

/**
 * data is interleaved if the interleave binding policy is set
 * or if pages are bound in a round-robin fashion.
 **/
int is_interleaved(void *data, const size_t size)
{
	return is_interleaved_policy(data) || is_interleaved_bind(data, size);
}

void *mmap_interleave_policy(const size_t size)
{
	const unsigned long NUMA_NODES[8] = { 1, 1, 1, 1, 1, 1, 1, 1 };
	void *out = mmap(NULL, size,
			 PROT_READ | PROT_WRITE,
			 MAP_PRIVATE | MAP_ANONYMOUS,
			 0, 0);

	if (out == NULL) {
		perror("mmap");
		exit(1);
	}

	if (mbind(out, size, MPOL_INTERLEAVE, NUMA_NODES,
		  sizeof(NUMA_NODES) / sizeof(*NUMA_NODES),
		  MPOL_MF_MOVE) == -1) {
		perror("mbind");
		munmap(out, size);
		exit(1);
	}

	return out;
}

void *mmap_interleave_bind(const size_t size)
{
	const int numnode = numa_max_node() == 0 ? 1 : numa_max_node();
	const unsigned long maxnode = sizeof(unsigned long) * 8;
	unsigned long NUMA_NODES = 1;
	int page_size = sysconf(_SC_PAGESIZE);
	intptr_t start;
	int node = 0;
	void *out;

	out = mmap(NULL, size, PROT_READ | PROT_WRITE,
		   MAP_PRIVATE | MAP_ANONYMOUS, 0, 0);

	if (out == NULL) {
		perror("mmap");
		exit(1);
	}

	start = ((intptr_t) out) << page_size >> page_size;
	for (intptr_t page = start; (size_t) (page - start) < size;
	     page += page_size) {

		NUMA_NODES = 1 << node;
		if (mbind((void *)page, page_size, MPOL_BIND, &NUMA_NODES,
			  maxnode, MPOL_MF_MOVE) == -1) {
			perror("mbind");
			munmap(out, size);
			exit(1);
		}
		node = (node + 1) % numnode;
	}

	return out;
}

int main(void)
{
	void *buf;
	const size_t size = (2 << 16);	// 16 pages

	// Check that linux bind interleave policy passes test.
	buf = mmap_interleave_policy(size);
	if (!is_interleaved(buf, size))
		return 1;
	printf("mmap_interleave_policy check works!\n");
	munmap(buf, size);

	// Check that binding pages in a round-robin fashion passes test.
	buf = mmap_interleave_bind(size);
	if (!is_interleaved(buf, size))
		return 1;
	printf("mmap_interleave_bind check works!\n");
	munmap(buf, size);

	return 0;
}
