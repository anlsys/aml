/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
*******************************************************************************/

#include <assert.h>
#include "aml.h"
#include <sys/mman.h>

/*******************************************************************************
 * Arena manager
 * Uses the local context (tid for example) to decide which arena will be used
 * for allocations.
 ******************************************************************************/

struct aml_arena * aml_area_linux_manager_single_get_arena(
				const struct aml_area_linux_manager_data *data)
{
	return data->pool;
}

struct aml_area_linux_manager_ops aml_area_linux_manager_single_ops = {
	aml_area_linux_manager_single_get_arena,
};

/*******************************************************************************
 * Initialization/Destroy function:
 ******************************************************************************/

int aml_area_linux_manager_single_init(struct aml_area_linux_manager_data *data,
				       struct aml_arena *arena)
{
	assert(data != NULL);
	data->pool = arena;
	data->pool_size = 1;
	return 0;
}

int aml_area_linux_manager_single_destroy(
				struct aml_area_linux_manager_data *data)
{
	assert(data != NULL);
	data->pool = NULL;
	return 0;
}


