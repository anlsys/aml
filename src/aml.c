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
#include "aml.h"

#ifdef HAVE_HWLOC
#include <hwloc.h>

hwloc_topology_t aml_topology = NULL;
size_t aml_pagesize = 4096;

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
		return err;
#endif
	aml_pagesize = sysconf(_SC_PAGE_SIZE);
	return 0;
	
}

int aml_finalize(void)
{
#ifdef HAVE_HWLOC
        hwloc_topology_destroy(aml_topology);
#endif	
	return 0;
}

