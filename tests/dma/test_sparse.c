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

#include "aml.h"

#include "aml/dma/linux-seq.h"
#include "aml/layout/sparse.h"
#include "aml/utils/features.h"
#if HAVE_CUDA != 0
#include "aml/area/cuda.h"
#include "aml/dma/cuda.h"
#include "aml/layout/cuda.h"
#endif

struct aml_layout *src, *cmp;

void setup(const size_t nptr)
{
	void *src_ptrs[nptr];
	void *cmp_ptrs[nptr];
	size_t sizes[nptr];
	size_t *src_data = malloc(nptr * sizeof(*src_data));
	size_t *cmp_data = malloc(nptr * sizeof(*cmp_data));

	assert(src_data != NULL);
	assert(cmp_data != NULL);

	for (size_t i = 0; i < nptr; i++) {
		src_data[i] = i;
		cmp_data[i] = 0;
		sizes[i] = sizeof(*sizes);
		src_ptrs[i] = src_data + i;
		cmp_ptrs[i] = cmp_data + i;
	}

	assert(aml_layout_sparse_create(&src, nptr, src_ptrs, sizes, NULL, 0) ==
	       AML_SUCCESS);
	assert(aml_layout_sparse_create(&cmp, nptr, cmp_ptrs, sizes, NULL, 0) ==
	       AML_SUCCESS);
}

void reset_cmp()
{
	struct aml_layout_sparse *scmp = (struct aml_layout_sparse *)cmp->data;
	for (size_t i = 0; i < scmp->nptr; i++)
		*(size_t *)(scmp->ptrs[i]) = 0;
}

void teardown()
{
	struct aml_layout_sparse *ssrc = (struct aml_layout_sparse *)src->data;
	struct aml_layout_sparse *scmp = (struct aml_layout_sparse *)cmp->data;
	free(ssrc->ptrs[0]);
	free(scmp->ptrs[0]);
	aml_layout_destroy(&src);
	aml_layout_destroy(&cmp);
}

// Test copy to dst then back to cmp and check that src and
// cmp hold the same bytes in their ptrs.
void test_dma_copy(struct aml_layout *src,
                   struct aml_layout *dst,
                   struct aml_layout *cmp,
                   struct aml_dma *src_dst,
                   struct aml_dma *dst_src,
                   aml_dma_operator op,
                   void *op_args)
{
	struct aml_layout_sparse *ssrc = (struct aml_layout_sparse *)src->data;
	struct aml_layout_sparse *scmp = (struct aml_layout_sparse *)cmp->data;

	assert(ssrc->nptr == scmp->nptr);
	assert(!memcmp(ssrc->sizes, scmp->sizes, ssrc->nptr * sizeof(size_t)));
	reset_cmp();
	assert(aml_dma_copy_custom(src_dst, dst, src, op, op_args) ==
	       AML_SUCCESS);
	assert(aml_dma_copy_custom(dst_src, cmp, dst, op, op_args) ==
	       AML_SUCCESS);
	for (size_t i = 0; i < ssrc->nptr; i++)
		assert(!memcmp(ssrc->ptrs[i], scmp->ptrs[i], ssrc->sizes[i]));
}

void test_dma_linux()
{
	struct aml_layout *dst;
	struct aml_dma *dma;
	struct aml_layout_sparse *ssrc = (struct aml_layout_sparse *)src->data;
	size_t data[ssrc->nptr];
	void *ptrs[ssrc->nptr];

	// Setup
	for (size_t i = 0; i < ssrc->nptr; i++)
		ptrs[i] = data + i;
	assert(aml_layout_sparse_create(&dst, ssrc->nptr, ptrs, ssrc->sizes,
	                                NULL, 0) == AML_SUCCESS);
	assert(aml_dma_linux_seq_create(&dma, 4, aml_layout_linux_copy_sparse,
	                                NULL) == AML_SUCCESS);

	// Test
	test_dma_copy(src, dst, cmp, dma, dma, aml_layout_linux_copy_sparse,
	              NULL);

	// Cleanup
	aml_layout_destroy(&dst);
	aml_dma_linux_seq_destroy(&dma);
}

#if HAVE_CUDA != 0
void test_dma_cuda()
{
	struct aml_layout *dst;
	struct aml_dma *dma_to_device, *dma_to_host;
	struct aml_layout_sparse *ssrc = (struct aml_layout_sparse *)src->data;
	int device_id = 0;
	void *ptrs[ssrc->nptr];

	// Setup
	for (size_t i = 0; i < ssrc->nptr; i++)
		ptrs[i] = aml_area_mmap(&aml_area_cuda, sizeof(size_t), NULL);
	assert(aml_layout_sparse_create(&dst, ssrc->nptr, ptrs, ssrc->sizes,
	                                &device_id,
	                                sizeof(device_id)) == AML_SUCCESS);
	assert(aml_dma_cuda_create(&dma_to_device, cudaMemcpyHostToDevice) ==
	       AML_SUCCESS);
	assert(aml_dma_cuda_create(&dma_to_host, cudaMemcpyDeviceToHost) ==
	       AML_SUCCESS);

	// Test
	test_dma_copy(src, dst, cmp, dma_to_device, dma_to_host,
	              aml_layout_cuda_copy_sparse, NULL);

	// Cleanup
	aml_layout_destroy(&dst);
	aml_dma_cuda_destroy(&dma_to_device);
	aml_dma_cuda_destroy(&dma_to_host);
	for (size_t i = 0; i < ssrc->nptr; i++)
		aml_area_munmap(&aml_area_cuda, ptrs[i], sizeof(size_t));
}
#endif

int main()
{
	setup(16);
	test_dma_linux();
#if HAVE_CUDA != 0
	if (aml_support_backends(AML_BACKEND_CUDA))
		test_dma_cuda();
#endif
	teardown();
	return 0;
}
