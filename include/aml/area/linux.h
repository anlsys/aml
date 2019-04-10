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
extern struct aml_area aml_area_linux;


/*******************************************************************************
 * Linux operators
*******************************************************************************/

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
aml_area_linux_munmap(const struct aml_area_data *area_data,
		      void *ptr,
		      const size_t size);

/*******************************************************************************
 * create/destroy and others
*******************************************************************************/


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
 * Allocate and initialize a struct aml_area implemented by aml_area_linux
 * operations.
 * @param[out] area pointer to an uninitialized struct aml_area pointer to
 * receive the new area.
 * @param[in] mmap_flags flags to use when retrieving virtual memory with mmap
 * @param[in] binding_flags, flags to use when binding memory.
 * @param[in] nodemask list of memory nodes to use. Default to allowed memory
 * nodes if NULL.
 * @return On success, returns 0 and area points to the new aml_area.
 * @return On failure, sets area to NULL and returns one of AML error codes:
 * - AML_ENOMEM if there wasn't enough memory available.
 * - AML_EINVAL if inputs flags were invalid.
 * - AML_EDOM the nodemask provided is out of bounds (allowed nodeset).
 **/
int aml_area_linux_create(struct aml_area **area, const int mmap_flags,
			  const struct aml_bitmap *nodemask,
			  const int binding_flags);

/**
 * Initialize a struct aml_area declared using the AML_AREA_LINUX_DECL macro.
 * See aml_area_linux_create for details on arguments.
 */
int aml_area_linux_init(struct aml_area *area, const int mmap_flags,
			const struct aml_bitmap *nodemask,
			const int binding_flags);
/**
 * Finalize a struct aml_area initialized with aml_area_linux_init.
 */
void aml_area_linux_fini(struct aml_area *area);

/**
 * Destroy (finalize and free resources) a struct aml_area created by
 * aml_area_linux_create.
 *
 * @param area is NULL after this call.
 **/
void aml_area_linux_destroy(struct aml_area **area);

#endif //AML_AREA_LINUX_NUMA_H
