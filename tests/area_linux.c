/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
*******************************************************************************/

#include <aml.h>
#include <assert.h>
#include <numa.h>
#include <numaif.h>

void doit(struct aml_area *area)
{
	void *ptr;
	unsigned long *a, *b, *c;
	intptr_t iptr;

	/* try to allocate something */
	ptr = aml_area_malloc(area, sizeof(unsigned long) * 10);
	assert(ptr != NULL);
	a = (unsigned long *)ptr;
	memset(a, 0, sizeof(unsigned long)*10);
	assert(a[0] == 0);
	assert(a[0] == a[9]);
	aml_area_free(area, ptr);

	/* libc API compatibility: malloc(0):
	 * returns either null or unique valid for free. */
	ptr = aml_area_malloc(area, 0);
	aml_area_free(area, ptr);

	/* calloc */
	ptr = aml_area_calloc(area, 10, sizeof(unsigned long));
	assert(ptr != NULL);
	a = (unsigned long *)ptr;
	assert(a[0] == 0);
	assert(a[0] == a[9]);
	aml_area_free(area, ptr);

	/* memalign */
	ptr = aml_area_memalign(area, 16, sizeof(unsigned long));
	assert(ptr != NULL);
	iptr = (intptr_t)ptr;
	assert(iptr % 16 == 0);
	aml_area_free(area, ptr);


	/* libc API compatibility: calloc(0): same as malloc(0) */
	ptr = aml_area_calloc(area, 0, sizeof(unsigned long));
	aml_area_free(area, ptr);
	ptr = aml_area_calloc(area, 10, 0);
	aml_area_free(area, ptr);

	/* realloc */
	ptr = aml_area_realloc(area, NULL, sizeof(unsigned long) * 10);
	assert(ptr != NULL);
	ptr = aml_area_realloc(area, ptr, sizeof(unsigned long) * 2);
	assert(ptr != NULL);
	ptr = aml_area_realloc(area, ptr, sizeof(unsigned long) * 20);
	assert(ptr != NULL);
	ptr = aml_area_realloc(area, ptr, 0);
}

#define ARRAY_SIZE(x) (sizeof(x)/sizeof(x[0]))

int main(int argc, char *argv[])
{
	AML_ARENA_JEMALLOC_DECL(arena);
	AML_AREA_LINUX_DECL(area);
	unsigned long nodemask[AML_NODEMASK_SZ];
	struct bitmask *allowed;

	/* library initialization */
	aml_init(&argc, &argv);

	/* init arguments */
	allowed = numa_get_mems_allowed();
	memcpy(nodemask, allowed->maskp, AML_NODEMASK_BYTES);
	assert(!aml_arena_jemalloc_init(&arena, AML_ARENA_JEMALLOC_TYPE_REGULAR));

	assert(!aml_area_linux_init(&area,
				    AML_AREA_LINUX_MANAGER_TYPE_SINGLE,
				    AML_AREA_LINUX_MBIND_TYPE_REGULAR,
				    AML_AREA_LINUX_MMAP_TYPE_ANONYMOUS,
				    &arena, MPOL_BIND, nodemask));

	doit(&area);

	/* same here, order matters. */
	assert(!aml_area_linux_destroy(&area));

	aml_finalize();
	return 0;
}
