/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *******************************************************************************/

#include "config.h"

#include <string.h>

#include "aml.h"

const int aml_version_major = AML_VERSION_MAJOR;
const int aml_version_minor = AML_VERSION_MINOR;
const int aml_version_patch = AML_VERSION_PATCH;
const char *aml_version_string = AML_VERSION_STRING;

int aml_errno;

#if HAVE_HWLOC == 1
#include <hwloc.h>

hwloc_topology_t aml_topology;
hwloc_const_bitmap_t allowed_nodeset;

int aml_topology_init(void)
{
	char *topology_input = getenv("AML_TOPOLOGY");

	if (hwloc_topology_init(&aml_topology) == -1)
		return -1;

	if (hwloc_topology_set_flags(
	            aml_topology,
	            HWLOC_TOPOLOGY_FLAG_THISSYSTEM_ALLOWED_RESOURCES |
	                    HWLOC_TOPOLOGY_FLAG_IS_THISSYSTEM) == -1)
		return -1;

	if (topology_input != NULL &&
	    hwloc_topology_set_xml(aml_topology, topology_input) == -1)
		return -1;

	if (hwloc_topology_load(aml_topology) == -1)
		return -1;
	return 0;
}
#endif

int aml_init(int *argc, char **argv[])
{
	// disable warnings
	(void)argc;
	(void)argv;

	// Initialize topology
#if HAVE_HWLOC == 1
	int err_hwloc;
	err_hwloc = aml_topology_init();
	if (err_hwloc < 0)
		return AML_FAILURE;
	allowed_nodeset = hwloc_topology_get_allowed_nodeset(aml_topology);
#endif

	return 0;
}

int aml_finalize(void)
{
	// Destroy topology
#if HAVE_HWLOC == 1
	hwloc_topology_destroy(aml_topology);
#endif
	return 0;
}
