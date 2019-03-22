/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
*******************************************************************************/

#ifndef AML_AREA_LINUX_NUMA_H
#define AML_AREA_LINUX_NUMA_H

#include <numa.h>

/* User data stored inside area */
struct aml_area_linux_binding{
	struct bitmask *nodeset; /* numanodes to use */
	int flags;               /* numaif.h mbind policy */
};

/* Bind memory on given nodeset with MPOL_BIND policy */
const extern unsigned long aml_area_linux_flag_bind;
/* Bind memory on given nodeset with MPOL_INTERLEAVE policy */
const extern unsigned long aml_area_linux_flag_interleave;
/* Bind memory on given nodeset with MPOL_PREFFERED policy */
const extern unsigned long aml_area_linux_flag_preferred;

/** Initialize area data with struct aml_area_linux_binding **/
int
aml_area_linux_create(struct aml_area* area);

/** Destroy area data containing struct aml_area_linux_binding **/
void
aml_area_linux_destroy(struct aml_area* area);

/** 
 * Function to set struct aml_area_linux_binding 
 * "binding": bitwise translation from struct aml_bitmap to struct bitmask.
 * "flags": use one of aml_area_linux_flag_*
 **/
int
aml_area_linux_bind(struct aml_area         *area,
	       const struct aml_bitmap *binding,
	       const unsigned long      flags);

/** Function to check whether binding of a ptr area match area settings. **/
int
aml_area_linux_check_binding(struct aml_area *area,
			void            *ptr,
			size_t           size);

/** Function call to aml_area_linux_mmap_generic() then bind the data. **/
int
aml_area_linux_mmap_mbind(const struct aml_area *area,
		     void                 **ptr,
		     size_t                 size,
		     int                    flags);

/** Function call to aml_area_linux_mmap_private() then bind the data. **/
int
aml_area_linux_mmap_private_mbind(const struct aml_area *area,
			     void                 **ptr,
			     size_t                 size);

/** Function call to aml_area_linux_mmap_shared() then bind the data. **/
int
aml_area_linux_mmap_shared_mbind(const struct aml_area *area,
			    void                 **ptr,
			    size_t                 size);

/** Function call to aml_area_linux_malloc() then bind the data. **/
int
aml_area_linux_malloc_mbind(const struct aml_area *area,
		       void                 **ptr,
		       size_t                 size,
		       size_t                 alignement);

#endif //AML_AREA_LINUX_NUMA_H
