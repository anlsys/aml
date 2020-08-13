/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include <aml.h>

#include <aml/area/linux.h> // Allocation of target data
#include <aml/dma/linux-seq.h> // dma from/to libc backend
#include <aml/layout/sparse.h> // Data layout

#define N 100

typedef struct myvec {
	size_t len;
	double *data;
} myvec_t;

//----------------------------------------------------------------------------//

/** Engine managing data movements. **/
struct aml_dma *dma = NULL;

/** Function used by the dma to perform copies **/
aml_dma_operator dma_op = aml_layout_linux_copy_sparse;

/** Create the dma engine **/
void setup()
{
	if (aml_dma_linux_seq_create(&dma, 1, dma_op, NULL) != AML_SUCCESS)
		exit(1);
}

void teardown()
{
	aml_dma_linux_seq_destroy(&dma);
}

//----------------------------------------------------------------------------//

void init(myvec_t *s)
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

	setup();
	target.len = N;
	target.data = aml_area_mmap(&aml_area_linux, size, NULL);
	aml_layout_sparse_create(&host_layout, 1, (void **)&host.data, &size,
	                         NULL, 0);
	aml_layout_sparse_create(&target_layout, 1, (void **)&target.data,
	                         &size, NULL, 0);

	init(&target);
	aml_dma_copy_custom(dma, host_layout, target_layout, dma_op, NULL);
	printf("s.data[%d]=%lf\n", N - 1, host.data[N - 1]);
	// s.data[99]=99.000000

	aml_area_munmap(&aml_area_linux, target.data,
	                host.len * sizeof(double));
	aml_layout_destroy(&host_layout);
	aml_layout_destroy(&target_layout);
	teardown();
}

// OpenMP Examples Version 5.0.0 - November 2019
