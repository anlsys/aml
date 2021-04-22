/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include <aml.h>
#include <aml/dma/linux-par.h>
#include <aml/layout/dense.h>

static inline void
CHK_ABORT(int err, const char *message)
{
	if (err != AML_SUCCESS) {
		fprintf(stderr, "%s: %s\n", message, aml_strerror(err));
		exit(1);
	}
}

int
main(void)
{
	int err;

	// The DMA
	struct aml_dma *dma;

	err = aml_dma_linux_par_create(&dma, 128, NULL, NULL);
	CHK_ABORT(err, "aml_dma_linux_par_create:");

	// The source data for the move.
	double src[8]      = {1, 2, 3, 4, 5, 6, 7, 8};
	size_t src_dims[1] = {8};
	struct aml_layout *src_layout;

	err = aml_layout_dense_create(&src_layout,
				      src,
				      AML_LAYOUT_ORDER_COLUMN_MAJOR,
				      sizeof(*src), // size of 1 element
				      1, // only 1 dimension: flat array
				      src_dims,
				      NULL,  // data is not strided
				      NULL); // data has no pitch.
	CHK_ABORT(err, "aml_layout_dense_create:");

	// The destination data for the move.
	double dst[8]      = {0, 0, 0, 0, 0, 0, 0, 0};
	size_t dst_dims[1] = {8};
	struct aml_layout *dst_layout;

	err = aml_layout_dense_create(&dst_layout,
				      dst,
				      AML_LAYOUT_ORDER_COLUMN_MAJOR,
				      sizeof(*dst),
				      1,
				      dst_dims,
				      NULL,
				      NULL);
	CHK_ABORT(err, "aml_layout_dense_create:");

	// Handle to the dma request we are about to issue.
	struct aml_dma_request *request;

	err = aml_dma_async_copy_custom(
	    dma, &request, dst_layout, src_layout, NULL, NULL);
	CHK_ABORT(err, "aml_dma_async_copy_custom:");

	// Wait request
	err = aml_dma_wait(dma, &request);

	// check results match.
	if (memcmp(src, dst, sizeof(src)))
		return 1;

	// cleanup
	aml_layout_destroy(&src_layout);
	aml_layout_destroy(&dst_layout);
	aml_dma_linux_par_destroy(&dma);

	return 0;
}
