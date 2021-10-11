/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *******************************************************************************/

#include "aml.h"
#if AML_HAVE_BACKEND_CUDA == 1
#include <cuda.h>
#include <cuda_runtime.h>
#endif
#if AML_HAVE_BACKEND_ZE == 1
#include <level_zero/ze_api.h>

#include "aml/utils/backend/ze.h"
#endif

int aml_dummy_memset(void *ptr, int val, size_t size)
{
	(void)ptr;
	(void)val;
	(void)size;
	return AML_SUCCESS;
}

int aml_linux_memset(void *ptr, int val, size_t size)
{
	memset(ptr, val, size);
	return AML_SUCCESS;
}

int aml_cuda_memset(void *ptr, int val, size_t size)
{
#if AML_HAVE_BACKEND_CUDA == 1
	assert(aml_support_backends(AML_BACKEND_CUDA));
	assert(cudaMemset(ptr, val, size) == cudaSuccess);
	return AML_SUCCESS;
#else
	(void)ptr;
	(void)val;
	(void)size;
	return -AML_ENOTSUP;
#endif
}

int aml_ze_memset(void *ptr, int val, size_t size)
{
#if AML_HAVE_BACKEND_ZE == 1
	/* ze_command_list_handle_t hCommandList; */
	/* ze_context_handle_t context; */
	/* ze_command_queue_desc_t command_queue_desc = { */
	/*         .stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES, */
	/*         .pNext = NULL, */
	/*         .ordinal = 0, */
	/*         .index = 0, */
	/*         .flags = ZE_COMMAND_LIST_FLAG_EXPLICIT_ONLY, */
	/*         .mode = ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS, */
	/*         .priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL, */
	/* }; */

	/* // Init context and commandlist. */
	/* assert(aml_ze_context_create(&context, */
	/*                              aml_ze_default_data->device[0]) == */
	/*        AML_SUCCESS); */

	/* assert(zeCommandListCreateImmediate( */
	/*                context, aml_ze_default_data->device[0], */
	/*                &command_queue_desc, */
	/*                &hCommandList) == ZE_RESULT_SUCCESS); */

	/* // Memset. */
	/* assert(zeCommandListAppendMemoryFill(hCommandList, ptr, &val, */
	/*                                      sizeof(val), size, NULL, 0, */
	/*                                      NULL) == ZE_RESULT_SUCCESS); */

	/* // Cleanup */
	/* zeCommandListDestroy(hCommandList); */
	/* assert(aml_ze_context_destroy(context) == AML_SUCCESS); */
	/* return AML_SUCCESS; */
	(void)ptr;
	(void)val;
	(void)size;
	return -AML_ENOTSUP;
#else
	(void)ptr;
	(void)val;
	(void)size;
	return -AML_ENOTSUP;
#endif
}
