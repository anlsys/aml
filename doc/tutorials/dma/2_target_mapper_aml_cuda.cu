/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

extern "C" {
#include <aml.h>
#include <aml/area/cuda.h> // Allocation of target data
#include <aml/dma/cuda.h> // dma from/to libc backend
#include <aml/layout/sparse.h> // Data layout
}

#define N 100

typedef struct myvec {
	size_t len;
	double *data;
} myvec_t;

__global__ void init(myvec_t *s)
{
	for (size_t i = 0; i < s->len; i++)
		s->data[i] = i;
}

int main()
{
	myvec_t host;

	host.data = (double *)calloc(N, sizeof(double));
	host.len = N;

	myvec_t target;
	size_t size = host.len * sizeof(double);
	struct aml_layout *host_layout, *target_layout;

	target.len = N;
	target.data = (double*)aml_area_mmap(&aml_area_cuda, size, NULL);
	aml_layout_sparse_create(&host_layout, 1, (void **)&host.data,
													 &size, NULL, 0);
	aml_layout_sparse_create(&target_layout, 1, (void **)&target.data,
	                         &size, NULL, 0);

	init<<<1,1>>>(&target);
	aml_dma_copy_custom(&aml_dma_cuda_device_to_host,
											host_layout, target_layout,
											aml_layout_cuda_copy_sparse, NULL);
	printf("s.data[%d]=%lf\n", N - 1, host.data[N - 1]);
	// s.data[99]=99.000000

	aml_area_munmap(&aml_area_cuda, target.data, size);
	aml_layout_destroy(&host_layout);
	aml_layout_destroy(&target_layout);
}

// OpenMP Examples Version 5.0.0 - November 2019
