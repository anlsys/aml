/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#ifndef AML_AREA_DUMMY_H
#define AML_AREA_DUMMY_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup aml_area_dummy "AML Dummy Areas"
 * @brief Implementation of AML areas doing nothing.
 *
 * This implementation of AML areas does not map/unmap any memory
 * and succeeds on all the calls to its methods.
 * The goal of this implementation is to measure the library overhead.
 *
 * @code
 * #include <aml/area/dummy.h>
 * @endcode
 * @{
 **/

/** Dummy area methods */
extern struct aml_area_ops aml_area_dummy_ops;

/** Dummy area. */
extern struct aml_area aml_area_dummy;

struct aml_area_dummy_data {
	size_t counter;
};

/**
 * `mmap()` method implementation for dummy area.
 * All parameters are ignored. The function returns a counter incremented by *
 * `size`.
 */
void *aml_area_dummy_mmap(const struct aml_area_data *area_data,
                          size_t size,
                          struct aml_area_mmap_options *opts);

/**
 * `munmap()` method implementation for dummy area.
 * All parameters are ignored. The function always returns AML_SUCCESS.
 */
int aml_area_dummy_munmap(const struct aml_area_data *area_data,
                          void *ptr,
                          const size_t size);

/**
 * @}
 **/

#ifdef __cplusplus
}
#endif
#endif // AML_AREA_DUMMY_H
