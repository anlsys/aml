/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include <mpi.h>
#include "aml.h"
#include "aml/area/mpi.h"

#include "internal/uthash.h"

/*******************************************************************************
 * Implementation
 ******************************************************************************/

struct aml_area_mpi_window {
	void *ptr;
	MPI_Win win;
	UT_hash_handle hh;
};

void *aml_area_mpi_mmap(const struct aml_area_data *area_data,
			 size_t size, struct aml_area_mmap_options *options)
{
	int err;
	struct aml_area_mpi_data *data;
	struct aml_area_mpi_mmap_options *opts;

	data = (struct aml_area_mpi_data *)area_data;
	opts = (struct aml_area_mpi_mmap_options *)options;

	MPI_Comm comm;
	MPI_Info info;
	int disp;

	if (opts != NULL) {
		comm = opts->comm;
		info = opts->info;
		disp = opts->disp;
	} else {
		comm = MPI_COMM_WORLD;
		MPI_Comm_get_info(comm, &info);
		disp = 1;
	}
	struct aml_area_mpi_window *w = calloc(1, sizeof(*w));
	assert(w != NULL);
	err = MPI_Win_allocate(size, disp, info, comm, &w->ptr, &w->win);
	/* TODO error conversion */
	if (err != MPI_SUCCESS) {
		free(w);
		aml_errno = -AML_FAILURE;
		return MAP_FAILED;
	}

	/* store the window for munmap */
	HASH_ADD_PTR(data->windows, ptr, w);
	if (opts != NULL) {
		opts->win = w->win;
	}
	return w->ptr;
}

int aml_area_mpi_munmap(const struct aml_area_data *area_data,
			 void *ptr, const size_t size)
{
	(void)size;
	struct aml_area_mpi_data *data;
	struct aml_area_mpi_window *w = NULL;

	data = (struct aml_area_mpi_data *)area_data;

	HASH_FIND_PTR(data->windows, &ptr, w);
	MPI_Win_free(&w->win);
	HASH_DEL(data->windows, w);
	free(w);
	return AML_SUCCESS;
}

/*******************************************************************************
 * Areas Initialization
 ******************************************************************************/

int aml_area_mpi_create(struct aml_area **area)
{
	struct aml_area *ret;
	struct aml_area_mpi_data *data;

	ret = AML_INNER_MALLOC(struct aml_area,
				      struct aml_area_mpi_data);
	if (ret == NULL)
		return -AML_ENOMEM;

	data = AML_INNER_MALLOC_GET_FIELD(ret, 2, struct aml_area,
					  struct aml_area_mpi_data);

	ret->ops = &aml_area_mpi_ops;
	ret->data = (struct aml_area_data *)data;

	*area = ret;
	return AML_SUCCESS;
}

void aml_area_mpi_destroy(struct aml_area **area)
{
	if (*area == NULL)
		return;

	free(*area);
	*area = NULL;
}

/*******************************************************************************
 * Areas declaration
 ******************************************************************************/

struct aml_area_mpi_data aml_area_mpi_data_default = {
	.windows = NULL,
};

struct aml_area_ops aml_area_mpi_ops = {
	.mmap = aml_area_mpi_mmap,
	.munmap = aml_area_mpi_munmap,
};

struct aml_area aml_area_mpi = {
	.ops = &aml_area_mpi_ops,
	.data = (struct aml_area_data *)(&aml_area_mpi_data_default)
};
