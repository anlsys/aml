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
#include <aml.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
/*******************************************************************************
 * mmap methods for Linux systems
 * Only handles the actual mmap call
 ******************************************************************************/

void *aml_area_linux_mmap_generic(struct aml_area_linux_mmap_data *data,
				  void *ptr, size_t sz)
{
	return mmap(ptr, sz, data->prot, data->flags, data->fildes, data->off);
}

struct aml_area_linux_mmap_ops aml_area_linux_mmap_generic_ops = {
	aml_area_linux_mmap_generic,
};

int aml_area_linux_mmap_anonymous_init(struct aml_area_linux_mmap_data *data)
{
	assert(data != NULL);
	data->prot = PROT_READ|PROT_WRITE;
	data->flags = MAP_PRIVATE|MAP_ANONYMOUS;
	data->fildes = -1;
	data->off = 0;
	return 0;
}

int aml_area_linux_mmap_anonymous_destroy(struct aml_area_linux_mmap_data *data)
{
	assert(data != NULL);
	return 0;
}

int aml_area_linux_mmap_fd_init(struct aml_area_linux_mmap_data *data, int fd,
				off_t offset)
{
	/* TODO: should we check for the right open flags ? */
	assert(data != NULL);
	data->prot = PROT_READ|PROT_WRITE;
	data->flags = MAP_PRIVATE;
	data->fildes = fd;
	data->off = offset;
	return 0;
}

/* doesn't close the fd on purpose */
int aml_area_linux_mmap_fd_destroy(struct aml_area_linux_mmap_data *data)
{
	assert(data != NULL);
	return 0;
}

int aml_area_linux_mmap_tmpfile_init(struct aml_area_linux_mmap_data *data,
				     char *template, size_t max)
{
	assert(data != NULL);
	data->prot = PROT_READ|PROT_WRITE;
	data->flags = MAP_PRIVATE;
	data->fildes = mkstemp(template);
	data->off = 0;
	int n = ftruncate(data->fildes, max);
	/* TODO: check & return errno */
	return 0;
}

int aml_area_linux_mmap_tmpfile_destroy(struct aml_area_linux_mmap_data *data)
{
	assert(data != NULL);
	return close(data->fildes);
}

