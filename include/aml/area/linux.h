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

/**
 * @defgroup aml_area_linux "AML Linux Areas"
 * @brief Linux Implementation of Areas.
 *
 * Linux implementation of AML areas.
 * This building block relies on libnuma implementation and
 * linux mmap/munmap to provide mmap/munmap on NUMA host
 * host processor memory. New areas may be created
 * to allocate a specific subset of memories.
 * This building block also include a static declaration of
 * a default initialized area that can be used out of the box with
 * abstract area API.
 *
 * #include <aml/area/linux.h>
 * @{
 **/

/**
 * This contains area operations implementation
 * for linux area.
 **/
extern struct aml_area_ops aml_area_linux_ops;

/**
 * Default linux area with private mapping and no binding.
 * Can be used out of the box with aml_area_*() functions.
 **/
extern struct aml_area aml_area_linux;

/** Allowed policy flag for area creation. **/
enum aml_area_linux_policy {
	/** default allocation policy **/
	AML_AREA_LINUX_POLICY_DEFAULT,
	/** Enforce binding on specified area nodeset or fail. **/
	AML_AREA_LINUX_POLICY_BIND,
	/**
	 * bind on specified area nodeset,
	 * fallback on other available nodes.
	 **/
	AML_AREA_LINUX_POLICY_PREFERRED,
	/** bind on specified area nodeset in a round-robin fashion. **/
	AML_AREA_LINUX_POLICY_INTERLEAVE,
};

/**
 * Implementation of aml_area_data for linux areas.
 **/
struct aml_area_linux_data {
	/** numanodes to use when allocating data **/
	struct bitmask *nodeset;
	/** binding policy **/
	enum aml_area_linux_policy policy;
};

/**
 * Options implementation for aml area linux mmap.
 * @see mmap man(2) page.
 **/
struct aml_area_linux_mmap_options {
	/** hint address where to perform allocation **/
	void *ptr;
	/** Combination of mmap flags **/
	int flags;
	/** prot flags **/
	int mode;
	/** File descriptor backing and initializing memory. **/
	int fd;
	/** Offset in file descriptor for mapping **/
	off_t offset;
};

/**
 * \brief Linux area creation.
 *
 * Allocate and initialize a struct aml_area implemented by aml_area_linux
 * operations.
 * @param[out] area pointer to an uninitialized struct aml_area pointer to
 *       receive the new area.
 * @param[in] nodemask list of memory nodes to use. Default to allowed memory
 *       nodes if NULL.
 * @param[in] policy: The memory allocation policy to use when binding on
 *       nodeset.
 * @return On success, returns 0 and area points to the new aml_area.
 * @return On failure, sets area to NULL and returns one of AML error codes:
 * - AML_ENOMEM if there wasn't enough memory available.
 * - AML_EINVAL if inputs flags were invalid.
 * - AML_EDOM the nodemask provided is out of bounds (allowed nodeset).
 **/
int aml_area_linux_create(struct aml_area **area,
			  const struct aml_bitmap *nodemask,
			  const enum aml_area_linux_policy policy);


/**
 * \brief Linux area destruction.
 *
 * Destroy (finalize and free resources) a struct aml_area created by
 * aml_area_linux_create().
 *
 * @param area is NULL after this call.
 **/
void aml_area_linux_destroy(struct aml_area **area);

/**
 * Bind memory of size "size" pointed by "ptr" to binding set in "bind".
 * If mbind call was not successfull, i.e AML_FAILURE is returned, then errno
 * should be inspected for further error checking.
 * @param bind: The binding settings. mmap_flags is actually unused.
 * @param ptr: The data to bind.
 * @param size: The size of the data pointed by ptr.
 * @return an AML error code.
 **/
int
aml_area_linux_mbind(struct aml_area_linux_data    *bind,
		     void                          *ptr,
		     size_t                         size);

/**
 * Function to check whether binding of a ptr obtained with
 * aml_area_linux_mmap() then aml_area_linux_mbind() match area settings.
 * @param area_data: The expected binding settings.
 * @param ptr: The data supposely bound.
 * @param size: The data size.
 * @return 1 if mapped memory binding in ptr match area_data binding settings,
 * else 0.
 **/
int
aml_area_linux_check_binding(struct aml_area_linux_data *area_data,
			     void                       *ptr,
			     size_t                      size);

/**
 * \brief mmap block for aml area.
 *
 * This function is a wrapper on mmap function using arguments set in
 * mmap_flags of area_data.
 * This function does not perform binding, unlike it is done in areas created
 * with aml_area_linux_create().
 * @param area_data: The structure containing mmap_flags for mmap call.
 *        nodemask and bind_flags fields are ignored.
 * @param size: The size to allocate.
 * @param opts: The size to allocate.
 * @return NULL on failure, else a valid pointer to memory.
 * Upon failure, errno should be checked for further error investigations.
 **/
void*
aml_area_linux_mmap(const struct aml_area_data  *area_data,
		    size_t                       size,
		    struct aml_area_mmap_options *opts);

/**
 * \brief munmap hook for aml area.
 *
 * unmap memory mapped with aml_area_linux_mmap().
 * @param area_data: unused
 * @param ptr: The virtual memory to unmap.
 * @param size: The size of virtual memory to unmap.
 * @return AML_FAILURE on error, AML_SUCCESS.
 * Upon failure errno should be checked for further error investigations.
 **/
int
aml_area_linux_munmap(const struct aml_area_data *area_data,
		      void *ptr,
		      const size_t size);

/**
 * @}
 **/

#endif //AML_AREA_LINUX_NUMA_H
