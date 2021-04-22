/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#ifndef AML_AREA_LINUX_NUMA_H
#define AML_AREA_LINUX_NUMA_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup aml_area_linux "AML Linux Areas"
 * @brief Linux Implementation of AML Areas.
 *
 * This building block relies on the libnuma implementation and
 * the Linux mmap() / munmap() to provide mmap() / munmap() on NUMA host
 * processor memory. New areas may be created
 * to allocate a specific subset of memories.
 * This building block also includes a static declaration of
 * a default initialized area that can be used out-of-the-box with
 * the abstract area API.
 *
 * @code
 * #include <aml/area/linux.h>
 * @endcode
 * @{
 **/

/**
 * Contains area operations implementation
 * for the Linux area.
 **/
extern struct aml_area_ops aml_area_linux_ops;

/**
 * Default Linux area using a private mapping and no binding.
 * Can be used out-of-the-box with aml_area_*() functions.
 **/
extern struct aml_area aml_area_linux;

/** Allowed policy flag for area creation. **/
enum aml_area_linux_policy {
	/** Default allocation policy. **/
	AML_AREA_LINUX_POLICY_DEFAULT,
	/**
	 * Enforce binding to the specified area nodeset; fail if not possible.
	 **/
	AML_AREA_LINUX_POLICY_BIND,
	/**
	 * Bind to the specified area nodeset;
	 * if not possible, fall back to other available nodes.
	 **/
	AML_AREA_LINUX_POLICY_PREFERRED,
	/** Bind to the specified area nodeset in a round-robin fashion. **/
	AML_AREA_LINUX_POLICY_INTERLEAVE,
};

/**
 * Implementation of aml_area_data for Linux areas.
 **/
struct aml_area_linux_data {
	/** numanodes to use when allocating data **/
	struct bitmask *nodeset;
	/** binding policy **/
	enum aml_area_linux_policy policy;
};

/**
 * Options implementation for aml_area_linux_mmap().
 * @see mmap(2) man page.
 **/
struct aml_area_linux_mmap_options {
	/** hint address where to perform allocation **/
	void *ptr;
	/** Combination of mmap flags **/
	int flags;
	/** protection flags **/
	int mode;
	/** File descriptor backing and initializing memory. **/
	int fd;
	/** Offset in the file for mapping **/
	off_t offset;
};

/**
 * \brief Linux area creation.
 *
 * Allocates and initializes struct aml_area implemented by aml_area_linux
 * operations.
 * @param[out] area pointer to an uninitialized struct aml_area pointer to
 *       receive the new area.
 * @param[in] nodemask list of memory nodes to use. Defaults to all allowed
 *       memory nodes if NULL.
 * @param[in] policy: The memory allocation policy to use when binding to
 *       nodeset.
 * @return On success, returns 0 and fills "area" with a pointer to the new
 *       aml_area.
 * @return On failure, fills "area" with NULL and returns one of AML error
 * codes:
 * - AML_ENOMEM if there wasn't enough memory available.
 * - AML_EINVAL if input flags were invalid.
 * - AML_EDOM if the nodemask provided was out of bounds (of the allowed
 *   node set).
 **/
int aml_area_linux_create(struct aml_area **area,
			  const struct aml_bitmap *nodemask,
			  const enum aml_area_linux_policy policy);


/**
 * \brief Linux area destruction.
 *
 * Destroys (finalizes and frees resources) struct aml_area created by
 * aml_area_linux_create().
 *
 * @param area address of an initialized struct aml_area pointer, which will be
 * reset to NULL on return from this call.
 **/
void aml_area_linux_destroy(struct aml_area **area);

/**
 * Binds memory of size "size" pointed to by "ptr" using the binding provided
 * in "bind". If the mbind() call was not successfull, i.e., AML_FAILURE is
 * returned, then "errno" should be inspected for further error information.
 * @param bind: The requested binding. "mmap_flags" is actually unused.
 * @param ptr: The memory to bind.
 * @param size: The size of the memory pointed to by "ptr".
 * @return 0 if successful; an error code otherwise.
 **/
int
aml_area_linux_mbind(struct aml_area_linux_data    *bind,
		     void                          *ptr,
		     size_t                         size);

/**
 * Checks whether the binding of a pointer obtained with
 * aml_area_linux_mmap() followed by aml_area_linux_mbind() matches the area
 * settings.
 * @param area_data: The expected binding settings.
 * @param ptr: The supposedly bound memory.
 * @param size: The memory size.
 * @return 1 if the mapped memory binding in "ptr" matches the "area_data"
 * binding settings, else 0.
 **/
int
aml_area_linux_check_binding(struct aml_area_linux_data *area_data,
			     void                       *ptr,
			     size_t                      size);

/**
 * \brief mmap block for AML area.
 *
 * This function is a wrapper around the mmap() call using arguments set in
 * "mmap_flags" of "area_data".
 * This function does not perform binding, unlike what is done in areas created
 * using aml_area_linux_create().
 * @param area_data: The structure containing "mmap_flags" for the mmap() call.
 *        "nodemask" and "bind_flags" fields are ignored.
 * @param size: The size to allocate.
 * @param opts: See "aml_area_linux_mmap_options".
 * @return a valid memory pointer, or NULL on failure.
 * On failure, "errno" should be checked for further information.
 **/
void*
aml_area_linux_mmap(const struct aml_area_data  *area_data,
		    size_t                       size,
		    struct aml_area_mmap_options *opts);

/**
 * \brief munmap hook for AML area.
 *
 * Unmaps memory mapped with aml_area_linux_mmap().
 * @param area_data: unused
 * @param ptr: The virtual memory to unmap.
 * @param size: The size of the virtual memory to unmap.
 * @return AML_SUCCESS on success, else AML_FAILURE.
 * On failure, "errno" should be checked for further information.
 **/
int
aml_area_linux_munmap(const struct aml_area_data *area_data,
		      void *ptr,
		      const size_t size);

/**
 * @}
 **/

#ifdef __cplusplus
}
#endif
#endif //AML_AREA_LINUX_NUMA_H
