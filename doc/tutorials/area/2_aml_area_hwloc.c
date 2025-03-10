/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
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

int max_bandwidth_area()
{
	int err;
	struct aml_area *area;

	// The object from which the bandwidth is maximized.
	hwloc_obj_t initiator =
	        hwloc_get_obj_by_type(aml_topology, HWLOC_OBJ_CORE, 0);

	// Area initialization
	err = aml_area_hwloc_preferred_create(
	        &area, initiator,
	        HWLOC_DISTANCES_KIND_FROM_OS |
	                HWLOC_DISTANCES_KIND_MEANS_LATENCY);

	if (err != AML_SUCCESS) {
		fprintf(stderr, "aml_area_hwloc_preferred_create: %s\n",
		        aml_strerror(-err));
		return 0; // Don't fail if the machine does not support this.
	}

	// Data allocated on fastest available memories from Core:0
	void *data = aml_area_mmap(area, size, NULL);

	if (data == NULL) {
		aml_perror("aml_area_mmap");
		aml_area_hwloc_preferred_destroy(&area);
		return 1;
	}

	// Cleanup
	aml_area_munmap(area, data, size);
	aml_area_hwloc_preferred_destroy(&area);
	return 0;
}

int main(int argc, char **argv)
{
	/* impossible to do those check in a CI environment consistently */
	if (!strcmp(getenv("CI"), "true"))
		exit(77);

	if (aml_init(&argc, &argv) != 0)
		return 1;

	if (static_interleave_area() != 0)
		goto err;

	if (max_bandwidth_area() != 0)
		goto err;

	// Cleanup success
	aml_finalize();
	return 0;

	// Cleanup error
err:
	aml_finalize();
	return 1;
}
