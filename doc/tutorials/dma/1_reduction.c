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
#include <aml/dma/linux-par.h>
#include <aml/layout/dense.h>

/** Error checking and printing. Should not be triggered **/
static inline void
CHK_ABORT(int err, const char *message)
{
	if (err != AML_SUCCESS) {
		fprintf(stderr, "%s: %s\n", message, aml_strerror(err));
		exit(1);
	}
}

/**
 * The ddot function
 **/
static inline double
ddot(const double *x, const double *y, const size_t n)
{
	double result = 0.0;

	for (size_t i = 0; i < n; i++)
		result += x[i] * y[i];
	return result;
}

int
main(void)
{
	int err;

	//---------------------------  Initialization
	//------------------------------
	struct aml_dma *dma;

	err = aml_dma_linux_par_create(&dma, 128, NULL, NULL);
	CHK_ABORT(err, "aml_dma_linux_par_create:");

	// Defining 'a' vector: {0.534, 65.4543, 0, 913.2} with a stride of 2.
	double a[8] = {
	    0.534, 6.3424, 65.4543, 4.543e12, 0.0, 1.0, 9.132e2, 23.657};
	size_t a_dims[1]   = {4}; // a has only 4 elements.
	size_t a_stride[1] = {2}; // elements are strided by 2.
	struct aml_layout *a_layout;

	err = aml_layout_dense_create(&a_layout,
				      a,
				      AML_LAYOUT_ORDER_COLUMN_MAJOR,
				      sizeof(*a),
				      1,
				      a_dims,
				      a_stride,
				      NULL);
	CHK_ABORT(err, "aml_layout_dense_create:");

	// Defining 'b' vector { 1.0, 1.0, 1.0, 1.0 } with a stride of 3.
	double b[12] = {
	    1.0,
	    0.0,
	    0.0,
	    1.0,
	    0.0,
	    0.0,
	    1.0,
	    0.0,
	    0.0,
	    1.0,
	    0.0,
	    0.0,
	};
	size_t b_dims[1]   = {4}; // b has 4 elements as well.
	size_t b_stride[1] = {3}; // b elements are strided by 3.
	struct aml_layout *b_layout;

	aml_layout_dense_create(&b_layout,
				b,
				AML_LAYOUT_ORDER_COLUMN_MAJOR,
				sizeof(*b),
				1,
				b_dims,
				b_stride,
				NULL);

	//-----------------  Defining ddot continous layouts
	//-----------------------

	double continuous_a[4];
	double continuous_b[4];
	size_t continuous_dims[1] = {4};
	struct aml_layout *a_continuous_layout;
	struct aml_layout *b_continuous_layout;

	err = aml_layout_dense_create(&a_continuous_layout,
				      continuous_a,
				      AML_LAYOUT_ORDER_COLUMN_MAJOR,
				      sizeof(*continuous_a),
				      1,
				      continuous_dims,
				      NULL,
				      NULL);
	CHK_ABORT(err, "aml_layout_dense_create:");

	err = aml_layout_dense_create(&b_continuous_layout,
				      continuous_b,
				      AML_LAYOUT_ORDER_COLUMN_MAJOR,
				      sizeof(*continuous_b),
				      1,
				      continuous_dims,
				      NULL,
				      NULL);
	CHK_ABORT(err, "aml_layout_dense_create:");

	//-----------------  Transform 'a' and 'b' to be continuous
	//-----------------

	// Handle to the dma request we are about to issue.
	struct aml_dma_request *a_request;
	struct aml_dma_request *b_request;

	// Schedule requests
	err =
	    aml_dma_async_copy(dma, &a_request, a_continuous_layout, a_layout);
	CHK_ABORT(err, "aml_dma_async_copy_custom:");

	err =
	    aml_dma_async_copy(dma, &b_request, b_continuous_layout, b_layout);
	CHK_ABORT(err, "aml_dma_async_copy_custom:");

	// Wait for the requests to complete
	err = aml_dma_wait(dma, &a_request);
	CHK_ABORT(err, "aml_dma_wait:");
	err = aml_dma_wait(dma, &b_request);
	CHK_ABORT(err, "aml_dma_wait:");

	//-----------------  Perform dot product and check result
	//-------------------

	double result = ddot(continuous_a, continuous_b, 4);

	// check results match.
	if (result != 979.1883)
		return 1;

	//----------------------------- Cleanup
	//-------------------------------------

	aml_layout_dense_destroy(&a_layout);
	aml_layout_dense_destroy(&b_layout);
	aml_layout_dense_destroy(&a_continuous_layout);
	aml_layout_dense_destroy(&b_continuous_layout);
	aml_dma_linux_par_destroy(&dma);

	return 0;
}
