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

#include <string.h>

#include "aml.h"

#include "aml/utils/backend/linux.h"
#if HAVE_HWLOC == 1
#include "aml/utils/backend/hwloc.h"
#endif
#if HAVE_MPI == 1
#include "aml/utils/backend/mpi.h"
#endif
#if HAVE_ZE == 1
#include "aml/utils/backend/ze.h"
#endif

const int aml_version_major = AML_VERSION_MAJOR;
const int aml_version_minor = AML_VERSION_MINOR;
const int aml_version_patch = AML_VERSION_PATCH;
const char *aml_version_revision = AML_VERSION_REVISION;
const char *aml_version_string = AML_VERSION_STRING;

int aml_errno;

int aml_init(int *argc, char **argv[])
{
	int err;

	// disable warnings
	(void)argc;
	(void)argv;

	err = aml_backend_linux_init();
	if (err != AML_SUCCESS)
		return err;

#if HAVE_MPI == 1
	err = aml_backend_mpi_init();
	if (err != AML_SUCCESS)
		goto err_with_linux;
#endif

#if HAVE_ZE == 1
	err = aml_backend_ze_init();
	if (err != AML_SUCCESS)
		goto err_with_mpi;
#endif

#if HAVE_HWLOC == 1
	err = aml_backend_hwloc_init();
	if (err != AML_SUCCESS)
		goto err_with_ze;
#endif

	return AML_SUCCESS;

// bit of ugly code here: labels can't be unused so we need to only define them
// if their caller is defined, but we only need to do something if the feature
// before them exists.
#if HAVE_HWLOC == 1
err_with_ze:
#endif
#if HAVE_ZE == 1
	aml_backend_ze_finalize();
#endif
#if HAVE_ZE == 1
err_with_mpi:
#endif
#if HAVE_MPI == 1
	aml_backend_mpi_finalize();
#endif
#if HAVE_MPI == 1
err_with_linux:
#endif
	aml_backend_linux_finalize();
	return err;
}

int aml_finalize(void)
{
	aml_backend_linux_finalize();
#if HAVE_HWLOC == 1
	aml_backend_hwloc_finalize();
#endif

#if HAVE_ZE == 1
	aml_backend_ze_finalize();
#endif

#if HAVE_MPI == 1
	aml_backend_mpi_finalize();
#endif
	return 0;
}
