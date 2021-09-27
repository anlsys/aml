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

#include "dummy_area.h"

struct aml_area_ops aml_area_dummy_ops = {.mmap = aml_area_dummy_mmap,
                                          .munmap = aml_area_dummy_munmap};

struct aml_area_dummy_data aml_area_dummy_data = {.counter = 1};
struct aml_area aml_area_dummy = {
        .data = (struct aml_area_data *)&aml_area_dummy_data,
        .ops = &aml_area_dummy_ops};

void *aml_area_dummy_mmap(const struct aml_area_data *area_data,
                          size_t size,
                          struct aml_area_mmap_options *opts)
{
	(void)opts;
	struct aml_area_dummy_data *data =
	        (struct aml_area_dummy_data *)area_data;

	if (data->counter + size < data->counter) {
		aml_errno = AML_ENOMEM;
		return NULL;
	}

	void *ptr = (void *)data->counter;
	data->counter += size;

	return ptr;
}

int aml_area_dummy_munmap(const struct aml_area_data *area_data,
                          void *ptr,
                          const size_t size)
{
	(void)area_data;
	(void)ptr;
	(void)size;
	return AML_SUCCESS;
}
