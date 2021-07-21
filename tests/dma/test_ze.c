/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include "aml.h"

#include "aml/area/ze.h"
#include "aml/dma/ze.h"
#include "aml/utils/backend/ze.h"
#define ZE(ze_call) aml_errno_from_ze_result(ze_call)

#include "test_dma.h"

int aml_memcpy_ze(struct aml_layout *dst,
                  const struct aml_layout *src,
                  void *arg)
{
	struct aml_dma_ze_copy_args *args = (struct aml_dma_ze_copy_args *)arg;
	size_t size = (size_t)args->arg;
	return ZE(zeCommandListAppendMemoryCopy(args->ze_data->command_list,
	                                        dst, src, size,
	                                        args->ze_req->event, 0, NULL));
}

int main(int argc, char **argv)
{
	assert(aml_init(&argc, &argv) == AML_SUCCESS);
	if (!aml_support_backends(AML_BACKEND_ZE))
		return 77;

	test_dma_memcpy(aml_area_ze_device, NULL, aml_dma_ze_default,
	                aml_memcpy_ze);

	aml_finalize();
}
