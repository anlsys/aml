/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <aml.h>
#include <aml/area/linux.h>
#include "tutorials.h"

void test_default_area(const size_t size)
{
	void *buf;

	buf = aml_area_mmap(&aml_area_linux, size, NULL);
	if (buf == NULL) {
		aml_perror("aml_area_linux");
		exit(1);
	}
	printf("Default linux area worked!\n");
	aml_area_munmap(&aml_area_linux, buf, size);
}

void test_interleave_area(const size_t size)
{
	int err;
	void *buf;
	struct aml_area *interleave_area;

	// Create interleave area on all nodes.
	err = aml_area_linux_create(&interleave_area,
				    NULL, AML_AREA_LINUX_POLICY_INTERLEAVE);
	if (err != AML_SUCCESS) {
		fprintf(stderr, "aml_area_linux_create: %s\n",
			aml_strerror(err));
		exit(1);
	}
	// Map buffer in area.
	buf = aml_area_mmap(interleave_area, size, NULL);
	if (buf == NULL) {
		aml_perror("aml_area_linux");
		exit(1);
	}
	// Check it is indeed interleaved
	if (!is_interleaved(buf, size))
		exit(1);
	printf("Interleave linux area worked and is interleaved.\n");

	// Cleanup
	aml_area_munmap(interleave_area, buf, size);
	aml_area_linux_destroy(&interleave_area);
}

int main(void)
{
	const size_t size = (2 << 16);	// 16 pages

	test_default_area(size);
	test_interleave_area(size);

	return 0;
}
