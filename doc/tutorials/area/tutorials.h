/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#ifndef AML_TUTOS_H
#define AML_TUTOS_H

#include <stdint.h>
#include <unistd.h>
#include <numaif.h>
#include <numa.h>

static int is_interleaved_policy(void *data)
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

static int get_node(void *data)
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

static int is_interleaved_bind(void *data, const size_t size)
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

static int is_interleaved(void *data, const size_t size)
{
	return is_interleaved_policy(data) || is_interleaved_bind(data, size);
}

#endif
