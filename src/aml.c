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

#include "aml/utils/linux.h"
#if HAVE_HWLOC == 1
#include "aml/utils/hwloc.h"
#endif
#if HAVE_ZE == 1
#include "aml/utils/ze.h"
#endif

const int aml_version_major = AML_VERSION_MAJOR;
const int aml_version_minor = AML_VERSION_MINOR;
const int aml_version_patch = AML_VERSION_PATCH;
const char *aml_version_string = AML_VERSION_STRING;

int aml_errno;

int aml_init(int *argc, char **argv[])
{
	int err;

	// disable warnings
	(void)argc;
	(void)argv;

	err = aml_linux_init();
	if (err != AML_SUCCESS)
		return err;

#if HAVE_ZE == 1
	err = aml_ze_init();
	if (err != AML_SUCCESS)
		goto error;
#endif

#if HAVE_HWLOC == 1
	err = aml_hwloc_init();
	if (err != AML_SUCCESS)
		goto err_with_ze;
#endif

	return AML_SUCCESS;

#if HAVE_HWLOC == 1
err_with_ze:;
#endif
#if HAVE_ZE == 1
	aml_ze_finalize();
error:
#endif
	aml_linux_finalize();
	return err;
}

int aml_finalize(void)
{
	aml_linux_finalize();
#if HAVE_HWLOC == 1
	aml_hwloc_finalize();
#endif

#if HAVE_ZE == 1
	aml_ze_finalize();
#endif
	return 0;
}
