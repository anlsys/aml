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
#include <mpi_runtime.h>
#include "aml.h"
#include "aml/area/mpi.h"

/*******************************************************************************
 * Implementation
 ******************************************************************************/

struct aml_mpi_area_mmap_header {
    MPI_Win win;        // the MPI window
    // void *baseptr;   // the local address of the window is implicit, it is right after the header
};

void *aml_area_mpi_mmap(const struct aml_area_data *area_data,
			 size_t size, struct aml_area_mmap_options *options)
{
    // TODO
    assert(options == NULL);

    struct aml_mpi_area_mmap_header header;
    MPI_Aint allocsize;
	struct aml_area_mpi_data *data;
    MPI_Win win;
    void * ptr;

    allocsize = sizeof(struct aml_mpi_area_mmap_header) + (MPI_Aint) size;
    d = (struct aml_area_mpi_data *)area_data;

    // args are:   in      in       in    in     out    out
    //           (size, disp_unit, info, comm, baseptr, win)
    if (MPI_Win_allocate(allocsize, 1, MPI_INFO_NULL, d->comm, (void *) &ptr, &header.win) != MPI_SUCCESS)
    {
        // TODO : could be
        //
        // MPI_ERR_ARG
        // Invalid argument. Some argument is invalid and is not identified by
        // a specific error class (e.g., MPI_ERR_RANK).
        //
        // MPI_ERR_COMM
        // Invalid communicator. A common error is to use a null communicator
        // in a call (not even allowed in MPI_Comm_rank).
        //
        // MPI_ERR_INFO
        // Invalid Info
        //
        // MPI_ERR_OTHER
        // Other error; use MPI_Error_string to get more information about this error code.
        //
        // MPI_ERR_SIZE

        aml_errno = -AML_FAILURE;
        return NULL;
    }

    // should be legal, as win2 = win1 is legal
    memcpy(ptr, &header, sizeof(aml_mpi_area_mmap_header));

	return ((char *)ptr) + sizeof(struct aml_mpi_area_mmap_header);
}

int aml_area_mpi_munmap(const struct aml_area_data *area_data,
			 void *ptr, const size_t size)
{
	(void)size;
	int flags = ((struct aml_area_mpi_data *)area_data)->flags;
	int error;

	// Unified Memory Allocation
	if (flags & AML_AREA_CUDA_FLAG_ALLOC_UNIFIED)
		error = mpiFree(ptr);
	// Mapped Allocation
	else if (flags & AML_AREA_CUDA_FLAG_ALLOC_MAPPED) {
		if (flags & AML_AREA_CUDA_FLAG_ALLOC_HOST)
			error = mpiFreeHost(ptr);
		else
			error = mpiHostUnregister(ptr);
	}
	// Host Allocation
	else if (flags & AML_AREA_CUDA_FLAG_ALLOC_HOST)
		error = mpiFreeHost(ptr);
	// Device Allocation
	else
		error = mpiFree(ptr);

	return mpi_to_aml_alloc_error(error);
}

int aml_area_mpi_fprintf(const struct aml_area_data *data,
			  FILE *stream, const char *prefix)
{
	const struct aml_area_mpi_data *d;
    int size, rank;
    char name[MPI_MAX_OBJECT_NAME];
    int name_len;

	fprintf(stream, "%s: area-mpi: %p\n", prefix, (void *)data);
	if (data == NULL)
		return AML_SUCCESS;

	d = (const struct aml_area_mpi_data *)data;

    if (MPI_Comm_size(comm, &size) != MPI_SUCCESS)
        size = -1;

    if (MPI_Comm_rank(comm, &rank) != MPI_SUCCESS)
        rank = -1;

    if (MPI_Comm_get_name(comm, name, &len) != MPI_SUCCESS)
        strncpy(name, "unkn", sizeof(name));

	fprintf(stream, "%s: comm-size: %d", prefix, size);
	fprintf(stream, "%s: comm-rank: %d", prefix, rank);
	fprintf(stream, "%s: comm-name: %*.s", prefix, name_len, name);

	return AML_SUCCESS;
}



/*******************************************************************************
 * Areas Initialization
 ******************************************************************************/

int aml_area_mpi_create(struct aml_area **area,
        MPI_Comm comm)
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

	data->comm = comm;

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
	.comm = MPI_COMM_WORLD
};

struct aml_area_ops aml_area_mpi_ops = {
	.mmap = aml_area_mpi_mmap,
	.munmap = aml_area_mpi_munmap,
	.fprintf = aml_area_mpi_fprintf,
};

struct aml_area aml_area_mpi = {
	.ops = &aml_area_mpi_ops,
	.data = (struct aml_area_data *)(&aml_area_mpi_data_default)
};
