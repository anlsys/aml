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

#include <sys/mman.h>
#include <numa.h>
#include <numaif.h>

/* Bind memory on given nodeset with MPOL_BIND policy */
#define AML_AREA_LINUX_BINDING_FLAG_BIND       (MPOL_BIND)
#define AML_AREA_LINUX_BINDING_FLAG_INTERLEAVE (MPOL_INTERLEAVE)
#define AML_AREA_LINUX_BINDING_FLAG_PREFERRED  (MPOL_PREFERRED)

/* Flags ti pass */
#define AML_AREA_LINUX_MMAP_FLAG_PRIVATE (MAP_PRIVATE)
#define AML_AREA_LINUX_MMAP_FLAG_SHARED  (MAP_SHARED)

/* User data stored inside area */
struct aml_area_linux_data {
	struct bitmask *nodeset;       /* numanodes to use */
	int             binding_flags; /* numaif.h mbind policy or AML_AREA_LINUX_FLAG_*/
	int             mmap_flags;    /* mmap flags */
};

/** Initialize area data with struct aml_area_linux_binding **/
struct aml_area* aml_area_linux_create(const int mmap_flags,
				       const struct aml_bitmap *nodemask,
				       const int binding_flags);


/** Destroy area data containing struct aml_area_linux_binding **/
void
aml_area_linux_destroy(struct aml_area* area);

/** Bind memory to area data. Done in aml_area_linux_mmap() **/
int
aml_area_linux_mbind(struct aml_area_linux_data    *bind,
		     void                          *ptr,
		     size_t                         size);
	
/** Function to check whether binding of a ptr area match area settings. **/
int
aml_area_linux_check_binding(struct aml_area_linux_data *area_data,
			     void                       *ptr,
			     size_t                      size);

/** Function call to aml_area_linux_mmap_generic() then bind the data. **/

void*
aml_area_linux_mmap(const struct aml_area_data  *area_data,
		    void                        *ptr,
		    size_t                       size);

/** Building block function for unmapping memory **/
int
aml_area_linux_munmap(const struct aml_area_data* area_data,
		      void *ptr,
		      const size_t size);


/* linux area hooks */
extern struct aml_area_ops aml_area_linux_ops;

/* Default linux area with private mapping and no binding. */
extern const struct aml_area aml_area_linux;

#endif //AML_AREA_LINUX_NUMA_H
	
