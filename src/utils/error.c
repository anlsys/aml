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

#include <stdio.h>

#include "aml.h"

#if HAVE_ZE == 1
#include <level_zero/ze_api.h>
#endif

static const char *const aml_error_strings[] = {
        [AML_SUCCESS] = "Success",
        [AML_FAILURE] = "Generic error",
        [AML_ENOMEM] = "Not enough memory",
        [AML_EINVAL] = "Invalid argument",
        [AML_EDOM] = "Value out of bound",
        [AML_EBUSY] = "Underlying resource is not available for operation",
        [AML_ENOTSUP] = "Operation not supported",
        [AML_EPERM] = "Insufficient permissions",
};

const char *aml_strerror(const int err)
{
	if (err < 0 || err >= AML_ERROR_MAX)
		return "Unknown error";
	return aml_error_strings[err];
}

void aml_perror(const char *msg)
{
	fprintf(stderr, "%s:%s\n", msg, aml_strerror(aml_errno));
}

#if HAVE_ZE == 1
int aml_errno_from_ze_result(ze_result_t err)
{
	switch (err) {
	case ZE_RESULT_SUCCESS:
		return AML_SUCCESS;
	case ZE_RESULT_ERROR_DEVICE_LOST:
	case ZE_RESULT_ERROR_MODULE_BUILD_FAILURE:
	case ZE_RESULT_ERROR_MODULE_LINK_FAILURE:
	case ZE_RESULT_ERROR_UNKNOWN:
	case ZE_RESULT_FORCE_UINT32:
		return AML_FAILURE;
	case ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY:
	case ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY:
		return AML_ENOMEM;
	case ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE:
	case ZE_RESULT_ERROR_UNINITIALIZED:
	case ZE_RESULT_ERROR_INVALID_ARGUMENT:
	case ZE_RESULT_ERROR_INVALID_NULL_HANDLE:
	case ZE_RESULT_ERROR_INVALID_NULL_POINTER:
	case ZE_RESULT_ERROR_INVALID_SIZE:
	case ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT:
	case ZE_RESULT_ERROR_INVALID_ENUMERATION:
	case ZE_RESULT_ERROR_INVALID_NATIVE_BINARY:
	case ZE_RESULT_ERROR_INVALID_GLOBAL_NAME:
	case ZE_RESULT_ERROR_INVALID_KERNEL_NAME:
	case ZE_RESULT_ERROR_INVALID_FUNCTION_NAME:
	case ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION:
	case ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION:
	case ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX:
	case ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE:
	case ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE:
	case ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED:
	case ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE:
		return AML_EINVAL;
	case ZE_RESULT_ERROR_OVERLAPPING_REGIONS:
		return AML_EDOM;
	case ZE_RESULT_ERROR_UNSUPPORTED_VERSION:
	case ZE_RESULT_ERROR_UNSUPPORTED_FEATURE:
	case ZE_RESULT_ERROR_UNSUPPORTED_SIZE:
	case ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT:
	case ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION:
	case ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT:
		return AML_ENOTSUP;
	case ZE_RESULT_NOT_READY:
	case ZE_RESULT_ERROR_NOT_AVAILABLE:
	case ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE:
		return AML_EBUSY;
	case ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS:
		return AML_EPERM;
	default:
		return AML_FAILURE;
	};
}
#endif
