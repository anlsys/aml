/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
*******************************************************************************/
#ifndef AML_AREA_H
#define AML_AREA_H

#include <aml/utils/bitmap.h>

/******************************************************************************
 * Lower level area management.
 * Implementations list:
 * - <aml/area/*.h>
 * - <aml/area/linux.h>
 ******************************************************************************/

/* Opaque handle to areas data. Defined by implementations */
struct aml_area_data;

/** Implementation specific operations. **/
struct aml_area_ops {
	/**
	 * Coarse grain allocator of virtual memory.
	 *
	 * "area_data": Opaque handle to implementation specific data.
	 * "ptr": A virtual address to be used by nderlying implementation. Can be NULL.
	 * "size": The minimum size of allocation. 
	 *         Is greater than 0. Must not fail unless not enough 
	 *         memory is available, or ptr argument does not point to a suitable address.
	 *         In case of failure, aml_errno must be set to an appropriate value.
	 *
	 * Returns a pointer to allocated memory object.
	 **/
        void* (*mmap)(const struct aml_area_data  *area_data,
		      void                        *ptr,
		      size_t                       size);
	
	/**
	 * Unmapping of virtual memory mapped with map().
	 *
	 * "area_data": An opaque handle to implementation specific data.
	 * "ptr": Pointer to data mapped in physical memory. Cannot be NULL.
	 * "size": The size of data. Cannot be 0.
	 *
	 * Returns AML_AREA_* error code.
	 **/
        int (*munmap)(const struct aml_area_data *area_data,
		      void                       *ptr,
		      size_t                      size);
	
};

struct aml_area {
	/* Basic memory operations implementation */
	struct aml_area_ops *ops;
	/* Implmentation specific data. Set to NULL at creation. */
	struct aml_area_data *data;
};

#endif //AML_AREA_H
