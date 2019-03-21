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
#include <string.h>

const char* aml_version_string = VERSION;
int         aml_major_version = -1;
int         aml_minor_version = -1;
int         aml_patch_version = -1;

int aml_get_major_version(){
	if(aml_major_version < 0){
		char version[strlen(VERSION)+1];
		strcpy(version, VERSION);
		return atoi(strtok(version, "."));
	}
	return aml_major_version;
}

int aml_get_minor_version(){
	if(aml_major_version < 0){
		char version[strlen(VERSION)+1];
		strcpy(version, VERSION);
		strtok(version, ".");
		return atoi(strtok(version, "."));
	}
	return aml_minor_version;
}

int aml_get_patch_version(){		
	if(aml_major_version < 0){
		char version[strlen(VERSION)+1];
		strcpy(version, VERSION);
		strtok(version, ".");
		strtok(version, ".");
		return atoi(strtok(version, "."));
	}
	return aml_minor_version;
}

#ifdef HAVE_HWLOC
#include <hwloc.h>

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
		return err;
#endif
	char version[strlen(VERSION)+1];
	strcpy(version, VERSION);
	aml_major_version = atoi(strtok(version, "."));
	if(aml_major_version != AML_ABI_VERSION)
		return -1;
	aml_minor_version = atoi(strtok(NULL, "."));
	aml_patch_version = atoi(strtok(NULL, "."));
	return 0;
	
}

int aml_finalize(void)
{
#ifdef HAVE_HWLOC
        hwloc_topology_destroy(aml_topology);
#endif	
	return 0;
}

