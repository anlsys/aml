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

/* allowed binding flags */
#define AML_AREA_LINUX_BINDING_FLAG_BIND       (MPOL_BIND)
#define AML_AREA_LINUX_BINDING_FLAG_INTERLEAVE (MPOL_INTERLEAVE)
#define AML_AREA_LINUX_BINDING_FLAG_PREFERRED  (MPOL_PREFERRED)

/* allowed mmap flags to pass */
#define AML_AREA_LINUX_MMAP_FLAG_PRIVATE (MAP_PRIVATE | MAP_ANONYMOUS)
#define AML_AREA_LINUX_MMAP_FLAG_SHARED  (MAP_SHARED | MAP_ANONYMOUS)

extern struct aml_area_ops aml_area_linux_ops;

/* User data stored inside area */
struct aml_area_linux_data {
	/** numanodes to use **/
	struct bitmask *nodeset;
	/** numaif.h mbind policy or AML_AREA_LINUX_FLAG_* **/
	int             binding_flags;
	/** mmap flags **/
	int             mmap_flags;
};

/* Default linux area with private mapping and no binding. */
extern const struct aml_area aml_area_linux;

#define AML_AREA_LINUX_DECL(name) \
	struct aml_area_linux_data __ ##name## _inner_data; \
	struct aml_area name = { \
		&aml_area_linux_ops, \
		(struct aml_area_data *)&__ ## name ## _inner_data, \
	}

#define AML_AREA_LINUX_ALLOCSIZE \
	(sizeof(struct aml_area_linux_data) + \
	 sizeof(struct aml_area))


/** 
 * Initialize area data with struct aml_area_linux_binding. Subsequent calls to
 * aml_area_mmap() with this returned area will apply binding settings.
 * Returns NULL on failure with aml_errno set to:
 * - AML_AREA_ENOMEM if there is not enough memory available for the operation
 * - AML_AREA_EINVAL flags were not one of linux area flags.
 * - AML_AREA_EDOM if binding nodeset is out of allowed nodeset.
 **/
struct aml_area* aml_area_linux_create(const int mmap_flags,
				       const struct aml_bitmap *nodemask,
				       const int binding_flags);


/** 
 * Destroy area data containing struct aml_area_linux_binding 
 **/
void
aml_area_linux_destroy(struct aml_area* area);

/**
 * Bind memory of size "size" pointed by "ptr" to binding set in "bind".
 * If mbind call was not successfull, i.e AML_FAILURE is returned, then errno
 * should be inspected for further error checking.
 **/
int
aml_area_linux_mbind(struct aml_area_linux_data    *bind,
		     void                          *ptr,
		     size_t                         size);

/** 
 * Function to check whether binding of a ptr obtained with 
 * aml_area_linux_mmap() then aml_area_linux_mbind() match area settings. 
 * Returns 1 if mapped memory binding in ptr match area_data binding settings, 
 * else 0.
 **/
int
aml_area_linux_check_binding(struct aml_area_linux_data *area_data,
			     void                       *ptr,
			     size_t                      size);

/** 
 * mmap hook for aml area.
 * Fails with AML_FAILURE. On failure errno should be checked for further
 * error investigations.
 **/
void*
aml_area_linux_mmap(const struct aml_area_data  *area_data,
		    void                        *ptr,
		    size_t                       size);

/** 
 * munmap hook for aml area, to unmap memory mapped with aml_area_linux_mmap().
 * Fails with AML_FAILURE. On failure errno should be checked for further
 * error investigations.
 **/
int
aml_area_linux_munmap(const struct aml_area_data* area_data,
		      void *ptr,
		      const size_t size);

#endif //AML_AREA_LINUX_NUMA_H
