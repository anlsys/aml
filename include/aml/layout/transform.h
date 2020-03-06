/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#ifndef AML_TRANSFORM_H
#define AML_TRANSFORM_H

/**
 * @defgroup aml_layout_transform "AML Layout Transform"
 * @brief Internal utils for performing transform on layouts.
 *
 * This module defines aml internal utils for performing layout transforms.
 *
 * Transforms are perfomed coordinate wise on two layouts with the same
 * number of elements. Starting from 0, coordinates are incremented by
 * one from the first dimension to the last and each element of source layout
 * is copied in destination layout.
 *
 * @code
 * #include <aml/layout/transform.h>
 * @endcode
 *
 * @{
 **/

/**
 * Function pointer for copying contiguous blocks of memory.
 * @return AML_SUCCESS or -AML_FAILURE.
 * If -AML_FAILURE is returned, aml_errno should be set with appropriate error.
 **/
typedef int (*aml_transform_memcpy_fn)(void*, const void*, size_t, void*);

/** Passed to aml_layout_transform for performing contiguous copies **/
struct aml_transform_args {
	aml_transform_memcpy_fn memcpy;
	void *data;
};

/**
 * @brief Generic Transform
 *
 * Function that can be passed as a aml_dma_operator to a dma implementation.
 * This function will walk every element of a layout and call args->memcpy
 * on every contiguous chunk of data according to the result of
 * aml_layout_deref_native(). Implementers of dma should implement
 * struct aml_transform_args to enjoy the convenience of this building block.
 **/
int aml_layout_transform(struct aml_layout *dst,
	const struct aml_layout *src,
	struct aml_transform_args *args);

/**
 * @brief Generic Copy
 *
 * Function that can be passed as a aml_dma_operator to a dma implementation.
 * This function will call once args->memcpy to perform a contiguous copy of
 * all src elements into dst regardless of the layout shape.
 **/
int
aml_layout_copy(struct aml_layout *dst,
	const struct aml_layout *src,
	struct aml_transform_args *args);

//------------------------------------------------------------------------------
// Pre Implemented Memcpy functors
//------------------------------------------------------------------------------

/** linux memcpy **/
int aml_linux_memcpy(void *dst, const void *src, size_t size, void *args);

/**
 * @}
 **/

#endif // AML_TRANSFORM_H
