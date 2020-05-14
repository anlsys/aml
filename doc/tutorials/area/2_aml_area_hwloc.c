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

#include <aml/area/hwloc.h>

//!\\ You should initialize your own topology instead. @see <hwloc.h>
extern hwloc_topology_t aml_topology;
static const size_t size = (1 << 22); // 4 MB

/**
 * Use static hwloc interleave area.
 **/
int static_interleave_area()
{
	void *data = aml_area_mmap(&aml_area_hwloc_interleave, size, NULL);

	// Worked ?
	if (data == NULL) {
		aml_perror("aml_area_mmap");
		return 1;
	}

	// Cleanup
	aml_area_munmap(&aml_area_hwloc_interleave, data, size);
	return 0;
}

int main(int argc, char **argv)
{
	if (aml_init(&argc, &argv) != 0)
		return 1;

	if (static_interleave_area() != 0)
		goto err;

	// Cleanup success
	aml_finalize();
	return 0;

	// Cleanup error
err:
	aml_finalize();
	return 1;
}
