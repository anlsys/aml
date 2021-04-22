/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *******************************************************************************/

#include "config.h"

#include <assert.h>
#include <fcntl.h>
#include <numa.h>
#include <numaif.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "aml.h"

#include "aml/area/linux.h"

int fd;

const size_t sizes[3] = {1, 1 << 12, 1 << 20};

struct aml_bitmap mems_allowed;
int last_node;

void test_area(struct aml_area *area, struct aml_area_mmap_options *options)
{
	void *ptr;

	for (size_t s = 0; s < sizeof(sizes) / sizeof(*sizes); s++) {
		ptr = aml_area_mmap(area, sizes[s], options);
		assert(ptr != NULL);
		memset(ptr, 1, sizes[s]);
		assert(aml_area_munmap(area, ptr, sizes[s]) == AML_SUCCESS);
	}
}

void test_case(const struct aml_bitmap *nodemask,
               const enum aml_area_linux_policy policy,
               const int flags)
{
	struct aml_area_linux_mmap_options options = {
	        .ptr = NULL,
	        .flags = MAP_ANONYMOUS | flags,
	        .mode = PROT_READ | PROT_WRITE,
	        .fd = -1,
	        .offset = 0,
	};

	struct aml_area *area;

	int err = aml_area_linux_create(&area, nodemask, policy);

	assert(err == AML_SUCCESS);

	// Map anonymous test.
	test_area(area, (struct aml_area_mmap_options *)(&options));

	// Map file test.
	options.flags = flags;
	options.fd = fd;
	test_area(area, (struct aml_area_mmap_options *)(&options));

	aml_area_linux_destroy(&area);
}

void test_policies(const struct aml_bitmap *nodemask, const int flags)
{
	test_case(nodemask, AML_AREA_LINUX_POLICY_BIND, flags);
	test_case(nodemask, AML_AREA_LINUX_POLICY_PREFERRED, flags);
	test_case(nodemask, AML_AREA_LINUX_POLICY_INTERLEAVE, flags);
}

void test_flags(const struct aml_bitmap *nodemask)
{
	test_policies(nodemask, MAP_SHARED);
	test_policies(nodemask, MAP_PRIVATE);
}

void test_single_node(void)
{
	struct aml_bitmap bitmap;

	aml_bitmap_zero(&bitmap);
	for (int i = 0; i <= last_node; i++) {
		if (aml_bitmap_isset(&mems_allowed, i)) {
			aml_bitmap_set(&bitmap, i);
			test_flags(&bitmap);
			aml_bitmap_clear(&bitmap, i);
		}
	}
}

void test_multiple_nodes(void)
{
	struct aml_bitmap bitmap;
	int first;

	aml_bitmap_zero(&bitmap);
	first = aml_bitmap_first(&mems_allowed);
	aml_bitmap_set(&bitmap, first);

	for (int i = first + 1; i <= last_node; i++) {
		if (aml_bitmap_isset(&mems_allowed, i)) {
			aml_bitmap_set(&bitmap, i);
			test_flags(&bitmap);
			aml_bitmap_clear(&bitmap, i);
		}
	}
}

int main(void)
{
	/* have to create the file in /tmp, otherwise we might end up running on
	 * a file system that doesn't like mmap */
	char tmp_name[] = "/tmp/test_area_linux_XXXXXX";
	size_t size = sizes[sizeof(sizes) / sizeof(*sizes) - 1];
	ssize_t nw = 0;
	char *buf;

	buf = malloc(size);
	assert(buf);
	memset(buf, 1, size);

	fd = mkstemp(tmp_name);
	assert(fd > 1);
	/* unlink right away to avoid leaving the file on the system after an
	 * error. The file will only disappear after close.
	 */
	unlink(tmp_name);

	nw = write(fd, buf, size);
	assert(nw == (ssize_t)size);
	free(buf);

	struct bitmask *nodeset = numa_get_mems_allowed();

	aml_bitmap_copy_from_ulong(&mems_allowed, nodeset->maskp,
	                           nodeset->size);
	last_node = aml_bitmap_last(&mems_allowed);

	test_single_node();
	test_multiple_nodes();

	close(fd);
	numa_bitmask_free(nodeset);
	return 0;
}
