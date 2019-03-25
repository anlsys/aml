/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
*******************************************************************************/

#include "aml.h"
#include "aml/utils/version.h"
#include "aml/utils/error.h"
#include "config.h"
#include <string.h>
#ifdef HAVE_HWLOC
#include <hwloc.h>
#endif

const int aml_version_major = AML_VERSION_MAJOR;
const int aml_version_minor = AML_VERSION_MINOR;
const int aml_version_patch = AML_VERSION_PATCH;
const char* aml_version_string = AML_VERSION_STRING;

int aml_errno = AML_SUCCESS;

#ifdef HAVE_HWLOC
hwloc_topology_t aml_topology = NULL;

int aml_topology_init(){
        if(hwloc_topology_init(&aml_topology) == -1)
		return -1;
	if(hwloc_topology_set_flags(aml_topology, HWLOC_TOPOLOGY_FLAG_THISSYSTEM_ALLOWED_RESOURCES) == -1)
		return -1;
	if(hwloc_topology_load(aml_topology) == -1)
		return -1;
	return 0;
}
#endif

int aml_init(int *argc, char **argv[])
{
#ifdef HAVE_HWLOC
	int err;
	
	err = aml_topology_init();
	if(err < 0)
		return AML_FAILURE;
#endif
	return 0;	
}

int aml_finalize(void)
{
#ifdef HAVE_HWLOC
        hwloc_topology_destroy(aml_topology);
#endif	
	return AML_SUCCESS;
}

