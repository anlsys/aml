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

#include "aml/dma/linux-par.h"
#include "aml/dma/linux-seq.h"

const int aml_version_major = AML_VERSION_MAJOR;
const int aml_version_minor = AML_VERSION_MINOR;
const int aml_version_patch = AML_VERSION_PATCH;
const char *aml_version_string = AML_VERSION_STRING;

int aml_errno;

struct aml_dma *aml_dma_linux_sequential;
struct aml_dma *aml_dma_linux_parallel;

#if HAVE_ZE == 1
#include "aml/area/ze.h"

struct aml_area *aml_area_ze_host;
struct aml_area *aml_area_ze_device;
#endif

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
	int err;

	// disable warnings
	(void)argc;
	(void)argv;

	// Initialize dma
	err = aml_dma_linux_seq_create(&aml_dma_linux_sequential, 64, NULL,
	                               NULL);
	if (err != AML_SUCCESS)
		goto err_with_linux_seq_dma;
	err = aml_dma_linux_par_create(&aml_dma_linux_parallel, 64, NULL, NULL);
	if (err != AML_SUCCESS)
		goto err_with_linux_par_dma;

		// Initialize level zero backend.
#if HAVE_ZE == 1
	// Test initializes the lib with zeInit().
	if (aml_support_backends(AML_BACKEND_ZE)) {
		err = aml_area_ze_create(&aml_area_ze_host,
		                         AML_AREA_ZE_MMAP_HOST_FLAGS,
		                         ZE_HOST_MEM_ALLOC_FLAG_BIAS_CACHED, 0,
		                         NULL, 64);
		if (err != ZE_RESULT_SUCCESS)
			goto err_with_linux_par_dma;
		err = aml_area_ze_create(&aml_area_ze_device,
		                         AML_AREA_ZE_MMAP_DEVICE_FLAGS, 0,
		                         ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_CACHED,
		                         NULL, 64);
		if (err != ZE_RESULT_SUCCESS) {
			aml_area_ze_destroy(&aml_area_ze_host);
			goto err_with_linux_par_dma;
		}
	} else {
		aml_area_ze_host = NULL;
		aml_area_ze_device = NULL;
	}
#endif

// Initialize topology
#if HAVE_HWLOC == 1
	int err_hwloc;
	err_hwloc = aml_topology_init();
	if (err_hwloc < 0)
		return AML_FAILURE;
	allowed_nodeset = hwloc_topology_get_allowed_nodeset(aml_topology);
#endif

	return 0;

err_with_linux_par_dma:
	aml_dma_linux_seq_destroy(&aml_dma_linux_sequential);
err_with_linux_seq_dma:
	return err;
}

int aml_finalize(void)
{
	aml_dma_linux_seq_destroy(&aml_dma_linux_sequential);
	aml_dma_linux_par_destroy(&aml_dma_linux_parallel);

	// Destroy topology
#if HAVE_HWLOC == 1
	hwloc_topology_destroy(aml_topology);
#endif

#if HAVE_ZE == 1
	if (aml_support_backends(AML_BACKEND_ZE)) {
		aml_area_ze_destroy(&aml_area_ze_host);
		aml_area_ze_destroy(&aml_area_ze_device);
	}
#endif
	return 0;
}
