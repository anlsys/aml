/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
*******************************************************************************/

#ifndef AML_H
#define AML_H 1

#include <assert.h>
#include <inttypes.h>
#include <numa.h>
#include <numaif.h>
#include <pthread.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

/* Used by bindings, specifically in aml_binding_nbpages() and
 * aml_binding_pages().  */
#ifndef PAGE_SIZE
#define PAGE_SIZE 4096
#endif

#include <aml/utils/bitmap.h>
#include <aml/utils/vector.h>


/*******************************************************************************
 * Forward Declarations:
 ******************************************************************************/

struct aml_area;
struct aml_binding;

/*******************************************************************************
 * Arenas:
 * In-memory allocator implementation. Dispatches actual memory mappings back to
 * areas.
 ******************************************************************************/

/* Arena Flags: access to useful jemalloc flags, with the same values. */
/* If passed as a flag to arena's mallocx()/reallocx() routines, the newly
 * allocated memory will be 0-initialized. */
#define AML_ARENA_FLAG_ZERO ((int)0x40)
/* As a flag to arena mallocx/reallocx, will align on (1<<a). */
#define AML_ARENA_FLAG_ALIGN(a) ((int)(a))

/* opaque handle to configuration data */
struct aml_arena_data;

struct aml_arena_ops {
	int (*register_arena)(struct aml_arena_data *arena,
			      struct aml_area *area);
	int (*deregister_arena)(struct aml_arena_data *arena);
	void *(*mallocx)(struct aml_arena_data *arena, size_t size, int flags);
	void (*dallocx)(struct aml_arena_data *arena, void *ptr, int flags);
	void *(*reallocx)(struct aml_arena_data *arena, void *ptr, size_t size,
			  int flags);
};

struct aml_arena {
	struct aml_arena_ops *ops;
	struct aml_arena_data *data;
};

/*
 * Registers a new memory arena with the system.  After this call the arena
 * is ready for use.
 * "arena": an initialized arena structure (see aml_arena_jemalloc_create()).
 * "area": a memory area that will be used as the backing store.
 * Returns 0 if successful; an error code otherwise.
 */
int aml_arena_register(struct aml_arena *arena, struct aml_area *area);
/*
 * Unregisters a memory arena from the system.  Also purges the contents of
 * the memory, so any buffers allocated from the arena should be considered
 * invalid after this call.
 * "arena": a registered arena structure (see aml_arena_register()).
 * Returns 0 if successful; an error code otherwise.
 */
int aml_arena_deregister(struct aml_arena *arena);
/*
 * Allocates a new memory buffer from the arena.
 * "arena": a registered arena structure (see aml_arena_register()).
 * "size": the buffer size in bytes; if 0 is passed, NULL will be returned.
 * "flags": see AML_ARENA_FLAG_*.
 * Returns a pointer to the newly allocated memory buffer; NULL if unsuccessful.
 */
void *aml_arena_mallocx(struct aml_arena *arena, size_t size, int flags);
/*
 * Releases a memory buffer back to the arena.
 * "arena": a registered arena structure (see aml_arena_register()).
 * "ptr": a pointer to the memory buffer or NULL (resulting in a no-op).
 * "flags": see AML_ARENA_FLAG_* (currently unused).
 */
void aml_arena_dallocx(struct aml_arena *arena, void *ptr, int flags);
/*
 * Changes the size of a previously allocated memory buffer.
 * "arena": a registered arena structure (see aml_arena_register()).
 * "ptr": a pointer to the memory buffer; if NULL is passed, acts just like
 *        aml_arena_mallocx().
 * "size": the new buffer size in bytes; if 0 is passed, acts just like
 *         aml_arena_dallocx() and returns NULL.
 * "flags": see AML_ARENA_FLAG_*.
 * Returns a pointer to the resized memory buffer; NULL if unsuccessful.
 */
void *aml_arena_reallocx(struct aml_arena *arena, void *ptr, size_t size,
			 int flags);

/*******************************************************************************
 * Jemalloc Arena:
 ******************************************************************************/
extern struct aml_arena_ops aml_arena_jemalloc_ops;

struct aml_arena_jemalloc_data {
	unsigned int uid;
	int flags;
};

#define AML_ARENA_JEMALLOC_DECL(name) \
	struct aml_arena_jemalloc_data __ ##name## _inner_data; \
	struct aml_arena name = { \
		&aml_arena_jemalloc_ops, \
		(struct aml_arena_data *)&__ ## name ## _inner_data, \
	};

#define AML_ARENA_JEMALLOC_ALLOCSIZE \
	(sizeof(struct aml_arena_jemalloc_data) + \
	 sizeof(struct aml_arena))

/* Arena types passed to jemalloc arena's create()/init()/vinit() routines.  */
/* Standard arena type.  */
#define AML_ARENA_JEMALLOC_TYPE_REGULAR 0
/* Arena type allocating memory-aligned buffers.  */
#define AML_ARENA_JEMALLOC_TYPE_ALIGNED 1
/* Arena type identical to an existing arena.  */
#define AML_ARENA_JEMALLOC_TYPE_GENERIC 2

/*
 * Allocates and initializes a new jemalloc arena.
 * "arena": an address where the pointer to the newly allocated arena structure
 *          will be stored.
 * "type": see AML_ARENA_JEMALLOC_TYPE_*.
 * Variadic arguments:
 * - if AML_ARENA_JEMALLOC_TYPE_REGULAR is passed as "type", no additional
 *   arguments are needed.
 * - if AML_ARENA_JEMALLOC_TYPE_ALIGNED is passed as "type", an alignment
 *   argument of type size_t and value that is a power of 2 must be provided.
 * - if AML_ARENA_JEMALLOC_TYPE_GENERIC is passed as "type", a pointer argument
 *   to type "aml_arena_data" (obtained from the "data" field of an existing
 *   jemalloc arena structure) must be provided.
 * Returns 0 if successful; an error code otherwise.
 */
int aml_arena_jemalloc_create(struct aml_arena **arena, int type, ...);
/*
 * Initializes a jemalloc arena.  This is a varargs-variant of the
 * aml_arena_jemalloc_vinit() routine.
 * "arena": an allocated jemalloc arena structure.
 * "type": see aml_arena_jemalloc_create().
 * Variadic arguments: see aml_arena_jemalloc_create().
 * Returns 0 if successful; an error code otherwise.
 */
int aml_arena_jemalloc_init(struct aml_arena *arena, int type, ...);
/*
 * Initializes a jemalloc arena.
 * "arena": an allocated jemalloc arena structure.
 * "type": see aml_arena_jemalloc_create().
 * "args": see the variadic arguments of aml_arena_jemalloc_create().
 * Returns 0 if successful; an error code otherwise.
 */
int aml_arena_jemalloc_vinit(struct aml_arena *arena, int type, va_list args);
/*
 * Tears down an initialized jemalloc arena.
 * "arena": an initialized jemalloc arena structure.
 * Returns 0 if successful; an error code otherwise.
 */
int aml_arena_jemalloc_destroy(struct aml_arena *arena);

/*******************************************************************************
 * Areas:
 * embeds information about a byte-addressable physical memory location as well
 * as binding policies over it.
 ******************************************************************************/

/* opaque handle to configuration data */
struct aml_area_data;

struct aml_area_ops {
	void *(*malloc)(struct aml_area_data *area, size_t size);
	void (*free)(struct aml_area_data *area, void *ptr);
	void *(*calloc)(struct aml_area_data *area, size_t num, size_t size);
	void *(*memalign)(struct aml_area_data *area, size_t align, size_t size);
	void *(*realloc)(struct aml_area_data *area, void *ptr, size_t size);
	void *(*acquire)(struct aml_area_data *area, size_t size);
	void (*release)(struct aml_area_data *area, void *ptr);
	void *(*mmap)(struct aml_area_data *area, void *ptr, size_t size);
	int (*available)(const struct aml_area_data *area);
	int (*binding)(const struct aml_area_data *area,
		       struct aml_binding **binding);
};

struct aml_area {
	struct aml_area_ops *ops;
	struct aml_area_data *data;
};

/*******************************************************************************
 * POSIX Area:
 ******************************************************************************/

extern struct aml_area_ops aml_area_posix_ops;

struct aml_area_posix_data {
};

#define AML_AREA_POSIX_DECL(name) \
	struct aml_area_posix_data __ ##name## _inner_data; \
	struct aml_area name = { \
		&aml_area_posix_ops, \
		(struct aml_area_data *)&__ ## name ## _inner_data, \
	};

#define AML_AREA_POSIX_ALLOCSIZE \
	(sizeof(struct aml_area_posix_data) + \
	 sizeof(struct aml_area))

/*
 * Allocates and initializes a new POSIX memory area.
 * "area": an address where the pointer to the newly allocated area structure
 *         will be stored.
 * Returns 0 if successful; an error code otherwise.
 */
int aml_area_posix_create(struct aml_area **area);
/*
 * Initializes a POSIX memory area.
 * "area": an allocated POSIX memory area structure.
 * Returns 0 if successful; an error code otherwise.
 */
int aml_area_posix_vinit(struct aml_area *area);
/*
 * Initializes a POSIX memory area.  This is identical to the
 * aml_area_posix_vinit() routine.
 * "area": an allocated POSIX memory area structure.
 * Returns 0 if successful; an error code otherwise.
 */
int aml_area_posix_init(struct aml_area *area);
/*
 * Tears down an initialized POSIX memory area.
 * "area": an initialized POSIX memory area structure.
 * Returns 0 if successful; an error code otherwise.
 */
int aml_area_posix_destroy(struct aml_area *area);

/*******************************************************************************
 * Linux Area:
 ******************************************************************************/

extern struct aml_area_ops aml_area_linux_ops;

struct aml_area_linux_manager_data {
	struct aml_arena *pool;
	size_t pool_size;
};

struct aml_area_linux_manager_ops {
	struct aml_arena *(*get_arena)(const struct aml_area_linux_manager_data *data);
};

extern struct aml_area_linux_manager_ops aml_area_linux_manager_single_ops;

/*
 * Initializes a Linux memory area manager.  A manager determines which arena
 * to use for allocations.
 * "data": an allocated Linux manager structure.
 * "arena": an arena to use for future allocations.
 * Returns 0 if successful; an error code otherwise.
 */
int aml_area_linux_manager_single_init(struct aml_area_linux_manager_data *data,
				       struct aml_arena *arena);
/*
 * Tears down an initialized Linux memory area manager.
 * "data": an initialized Linux manager structure.
 * Returns 0 if successful; an error code otherwise.
 */
int aml_area_linux_manager_single_destroy(struct aml_area_linux_manager_data *data);

struct aml_area_linux_mbind_data {
	struct aml_bitmap nodemask;
	int policy;
};

struct aml_area_linux_mbind_ops {
	int (*pre_bind)(struct aml_area_linux_mbind_data *data);
	int (*post_bind)(struct aml_area_linux_mbind_data *data, void *ptr,
			 size_t size);
	int (*binding)(const struct aml_area_linux_mbind_data *data,
		       struct aml_binding **binding);
};

/*
 * Sets memory policy of a Linux memory area.
 * "data": an initialized Linux memory policy structure.
 * "policy", "nodemask": see aml_area_linux_mbind_init().
 * Returns 0 if successful; an error code otherwise.
 */
int aml_area_linux_mbind_setdata(struct aml_area_linux_mbind_data *data,
				 int policy, const struct aml_bitmap *nodemask);
/*
 * Creates a new binding structure based on an existing Linux memory policy
 * structure.
 * "data": an initialized Linux memory policy structure.
 * "binding": an address where the pointer to the newly allocated binding
 *            structure will be stored.
 * Returns 0 if successful; an error code otherwise.
 */
int aml_area_linux_mbind_generic_binding(const struct aml_area_linux_mbind_data *data,
					 struct aml_binding **binding);
/*
 * Sets current memory policy before memory allocation using the Linux memory
 * area.
 * This variant is used with AML_AREA_LINUX_MBIND_TYPE_REGULAR mbind type (see
 * aml_area_linux_create()) and is basically a no-op.
 * "data": an initialized Linux memory policy structure.
 * Returns 0 if successful; an error code otherwise.
 */
int aml_area_linux_mbind_regular_pre_bind(struct aml_area_linux_mbind_data *data);
/*
 * Sets current memory policy on a new memory region allocated using the Linux
 * memory area.
 * This variant is used with AML_AREA_LINUX_MBIND_TYPE_REGULAR mbind type (see
 * aml_area_linux_create()).
 * "data": an initialized Linux memory policy structure.
 * "ptr": an address of the newly allocated memory region.
 * "size": the size of the newly allocated memory region.
 * Returns 0 if successful; an error code otherwise.
 */
int aml_area_linux_mbind_regular_post_bind(struct aml_area_linux_mbind_data *data,
					   void *ptr, size_t size);
/*
 * Sets current memory policy before memory allocation using the Linux memory
 * area.
 * This variant is used with AML_AREA_LINUX_MBIND_TYPE_MEMPOLICY mbind type (see
 * aml_area_linux_create()).
 * "data": an initialized Linux memory policy structure.
 * Returns 0 if successful; an error code otherwise.
 */
int aml_area_linux_mbind_mempolicy_pre_bind(struct aml_area_linux_mbind_data *data);
/*
 * Sets current memory policy on a new memory region allocated using the Linux
 * memory area.
 * This variant is used with AML_AREA_LINUX_MBIND_TYPE_MEMPOLICY mbind type (see
 * aml_area_linux_create()).
 * "data": an initialized Linux memory policy structure.
 * "ptr": an address of the newly allocated memory region.
 * "size": the size of the newly allocated memory region.
 * Returns 0 if successful; an error code otherwise.
 */
int aml_area_linux_mbind_mempolicy_post_bind(struct aml_area_linux_mbind_data *data,
					   void *ptr, size_t size);
/*
 * Initializes memory policy of a Linux memory area.
 * "data": an allocated Linux memory policy structure.
 * "policy": see MPOL_* in mbind(2).
 * "nodemask": an AML_MAX_NUMA_NODES-bit array (a AML_NODEMASK_SZ-element array)
 *             containing the NUMA node mask to use (see mbind(2) for more
 *             information).
 * Returns 0 if successful; an error code otherwise.
 */
int aml_area_linux_mbind_init(struct aml_area_linux_mbind_data *data,
			      int policy, const struct aml_bitmap *nodemask);
/*
 * Tears down an initialized Linx memory policy.
 * "data": an initialized Linux memory policy structure.
 * Returns 0 if successful; an error code otherwise.
 */
int aml_area_linux_mbind_destroy(struct aml_area_linux_mbind_data *data);

extern struct aml_area_linux_mbind_ops aml_area_linux_mbind_regular_ops;
extern struct aml_area_linux_mbind_ops aml_area_linux_mbind_mempolicy_ops;

struct aml_area_linux_mmap_data {
	int prot;
	int flags;
	int fildes;
	off_t off;
};

struct aml_area_linux_mmap_ops {
	void *(*mmap)(struct aml_area_linux_mmap_data *data, void *ptr,
		      size_t size);
};

/*
 * Allocates a memory region from a Linux memory area.
 * "data": an initialized Linux memory map structure.
 * "ptr": an address where the new memory region should be allocated (hint only;
 *        can be NULL to let the kernel decide).
 * "size": the requested size of thew new memory region to allocate.
 * Returns the address of the newly allocated region or MAP_FAILED (see mmap(2))
 * if unsuccessful.
 */
void *aml_area_linux_mmap_generic(struct aml_area_linux_mmap_data *data,
				  void *ptr, size_t size);
/*
 * Initializes memory map of a Linux memory area to use an anonymous
 * (0-initialized) mapping.
 * This variant is used with AML_AREA_LINUX_MMAP_TYPE_ANONYMOUS map type (see
 * aml_area_linux_create()).
 * "data": an allocated Linux memory map structure.
 * Returns 0 if successful; an error code otherwise.
 */
int aml_area_linux_mmap_anonymous_init(struct aml_area_linux_mmap_data *data);
/*
 * Initializes memory map of a Linux memory area to use an existing file
 * mapping.
 * This variant is used with AML_AREA_LINUX_MMAP_TYPE_FD map type (see
 * aml_area_linux_create()).
 * "data": an allocated Linux memory map structure.
 * "fd": an open file descriptor.
 * "offset": the offset within the file to allocate from.
 * Returns 0 if successful; an error code otherwise.
 */
int aml_area_linux_mmap_fd_init(struct aml_area_linux_mmap_data *data, int fd,
				off_t offset);
/*
 * Initializes memory map of a Linux memory area to use a newly created,
 * temporary file mapping.
 * This variant is used with AML_AREA_LINUX_MMAP_TYPE_TMPFILE map type (see
 * aml_area_linux_create()).
 * "data": an allocated Linux memory map structure.
 * "template": a file name template, ending in "XXXXXX"; the last six characters
 *             will be replaced with the actual name on successful file creation
 *             (see mkstemp(3) for more information).
 * "max": the size of the temporary file to create.
 * Returns 0 if successful; an error code otherwise.
 */
int aml_area_linux_mmap_tmpfile_init(struct aml_area_linux_mmap_data *data,
				     char *template, size_t max);
/*
 * Tears down an initialized Linux memory map.
 * This variant is used with AML_AREA_LINUX_MMAP_TYPE_ANONYMOUS map type (see
 * aml_area_linux_create()).
 * "data": an initialized Linux memory map structure.
 * Returns 0 if successful; an error code otherwise.
 */
int aml_area_linux_mmap_anonymous_destroy(struct aml_area_linux_mmap_data *data);
/*
 * Tears down an initialized Linux memory map.
 * This variant is used with AML_AREA_LINUX_MMAP_TYPE_FD map type (see
 * aml_area_linux_create()).
 * "data": an initialized Linux memory map structure.
 * Returns 0 if successful; an error code otherwise.
 */
int aml_area_linux_mmap_fd_destroy(struct aml_area_linux_mmap_data *data);
/*
 * Tears down an initialized Linux memory map.
 * This variant is used with AML_AREA_LINUX_MMAP_TYPE_TMPFILE map type (see
 * aml_area_linux_create()).
 * "data": an initialized Linux memory map structure.
 * Returns 0 if successful; an error code otherwise.
 */
int aml_area_linux_mmap_tmpfile_destroy(struct aml_area_linux_mmap_data *data);

extern struct aml_area_linux_mmap_ops aml_area_linux_mmap_generic_ops;

struct aml_area_linux_data {
	struct aml_area_linux_manager_data manager;
	struct aml_area_linux_mbind_data mbind;
	struct aml_area_linux_mmap_data mmap;
};

struct aml_area_linux_ops {
	struct aml_area_linux_manager_ops manager;
	struct aml_area_linux_mbind_ops mbind;
	struct aml_area_linux_mmap_ops mmap;
};

struct aml_area_linux {
	struct aml_area_linux_data data;
	struct aml_area_linux_ops ops;
};

#define AML_AREA_LINUX_DECL(name) \
	struct aml_area_linux __ ##name## _inner_data; \
	struct aml_area name = { \
		&aml_area_linux_ops, \
		(struct aml_area_data *)&__ ## name ## _inner_data, \
	};

#define AML_AREA_LINUX_ALLOCSIZE \
	(sizeof(struct aml_area_linux) + \
	 sizeof(struct aml_area))

/* Linux memory area manager types, passed to Linux memory area's
   create()/init()/vinit() routines.  */
/* Single-arena manager.  */
#define AML_AREA_LINUX_MANAGER_TYPE_SINGLE 0

/* Linux memory area mbind types, passed to Linux memory area's
   create()/init()/vinit() routines.  */
/* Regular type using mbind() after mmap().  */
#define AML_AREA_LINUX_MBIND_TYPE_REGULAR 0
/* Calls set_mempolicy() before and after mmap() to change the memory policy
   globally.  */
#define AML_AREA_LINUX_MBIND_TYPE_MEMPOLICY 1

/* Linux memory area map types, passed to Linux memory area's
   create()/init()/vinit() routines.  */
/* Zero-initialized, anonymous mapping.  */
#define AML_AREA_LINUX_MMAP_TYPE_ANONYMOUS 0
/* Mapping using an existing file.  */
#define AML_AREA_LINUX_MMAP_TYPE_FD 1
/* Mapping using a newly created temporary file.  */
#define AML_AREA_LINUX_MMAP_TYPE_TMPFILE 2

/*
 * Allocates and initializes a new Linux memory area.
 * "area": an address where the pointer to the newly allocated Linux memory area
 *         will be stored.
 * "manager_type": see AML_AREA_LINUX_MANAGER_TYPE_*.
 * "mbind_type": see AML_AREA_LINUX_MBIND_TYPE_*.
 * "mmap_type": see AML_AREA_LINUX_MMAP_TYPE_*.
 * Variadic arguments:
 * - "policy": an argument of type int; see aml_area_linux_mbind_init().
 * - "nodemask": an argument of type const unsigned long*;
 *               see aml_area_linux_mbind_init().
 * - if AML_AREA_LINUX_MMAP_TYPE_ANONYMOUS is passed as "mmap_type", no
 *   additional arguments are needed.
 * - if AML_AREA_LINUX_MMAP_TYPE_FD is passed as "mmap_type", two additional
 *   arguments are needed:
 *   - "fd": an argument of type int; see aml_area_linux_mmap_fd_init().
 *   - "offset": an argument of type off_t; see aml_area_linux_mmap_fd_init().
 * - if AML_AREA_LINUX_MMAP_TYPE_TMPFILE is passed as "mmap_type", two
 *   additional arguments are needed:
 *   - template: an argument of type char*; see
 *     aml_area_linux_mmap_tmpfile_init().
 *   - max: an argument of type size_t; see aml_area_linux_mmap_tmpfile_init().
 * Returns 0 if successful; an error code otherwise.
 */
int aml_area_linux_create(struct aml_area **area, int manager_type,
			  int mbind_type, int mmap_type, ...);
/*
 * Initializes a Linux memory area.  This is a varargs-variant of the
 * aml_area_linux_vinit() routine.
 * "area": an allocated Linux memory area structure.
 * "manager_type", "mbind_type", "mmap_type": see aml_area_linux_create().
 * Variadic arguments: see aml_area_linux_create().
 * Returns 0 if successful; an error code otherwise.
 */
int aml_area_linux_init(struct aml_area *area, int manager_type, int mbind_type,
			int mmap_type, ...);
/*
 * Initializes a Linux memory area.
 * "area": an allocated Linux memory area structure.
 * "manager_type", "mbind_type", "mmap_type": see aml_area_linux_create().
 * "args": see the variadic arguments of aml_area_linux_create().
 * Returns 0 if successful; an error code otherwise.
 */
int aml_area_linux_vinit(struct aml_area *area, int manager_type,
			 int mbind_type, int mmap_type, va_list args);
/*
 * Tears down an initialized Linux memory area.
 * "area": an initialized Linux memory area structure.
 * Returns 0 if successful; an error code otherwise.
 */
int aml_area_linux_destroy(struct aml_area *area);

/*******************************************************************************
 * Generic Area API:
 * Low-level, direct access to area logic.
 * For memory allocation function, follows the POSIX spec.
 ******************************************************************************/

/*
 * Allocates a new memory buffer from a memory area.
 * "area": an initialized memory area structure.
 * "size": the buffer size in bytes; if 0 is passed, NULL will be returned.
 * Returns a pointer to the newly allocated memory buffer; NULL if unsuccessful.
 */
void *aml_area_malloc(struct aml_area *area, size_t size);
/*
 * Releases a memory buffer back to the memory area.
 * "area": an initialized memory area structure.
 * "ptr": a pointer to the memory buffer or NULL (resulting in a no-op).
 */
void aml_area_free(struct aml_area *area, void *ptr);
/*
 * Allocates a new, zero-initialized memory buffer from a memory area.
 * "area": an initialized memory area structure.
 * "num": the number of elements of size "size" to allocate; if 0 is passed,
 *        NULL will be returned
 * "size": the size of each individual element to allocate, in bytes; if 0 is
 *         passed, NULL will be returned.
 * Returns a pointer to the newly allocated memory buffer; NULL if unsuccessful.
 */
void *aml_area_calloc(struct aml_area *area, size_t num, size_t size);
/*
 * Allocates a new, aligned, memory buffer from a memory area.
 * "area": an initialized memory area structure.
 * "align": an alignment for the returned pointer.
 * "size": the buffer size in bytes; if 0 is passed, NULL will be returned.
 * Returns a pointer to the newly allocated memory buffer; NULL if unsuccessful.
 */
void *aml_area_memalign(struct aml_area *area, size_t align, size_t size);
/*
 * Changes the size of a previously allocated memory buffer.
 * "area": an initialized memory area structure.
 * "ptr": a pointer to the memory buffer; if NULL is passed, acts just like
 *        aml_area_malloc().
 * "size": the new buffer size in bytes; if 0 is passed, acts just like
 *         aml_area_free() and returns NULL.
 * Returns a pointer to the resized memory buffer; NULL if unsuccessful.
 */
void *aml_area_realloc(struct aml_area *area, void *ptr, size_t size);
/* FIXME! */
void *aml_area_acquire(struct aml_area *area, size_t size);
/* FIXME! */
void aml_area_release(struct aml_area *area, void *ptr);
/*
 * Allocates a memory region from a Linux memory area, respecting memory policy
 * settings (see aml_area_linux_mbind_init()).
 * "area": an initialized memory area structure.
 * "ptr": an address where the new memory region should be allocated (hint only;
 *        can be NULL to let the kernel decide).
 * "size": the requested size of thew new memory region to allocate.
 * Returns the address of the newly allocated region or MAP_FAILED (see mmap(2))
 * if unsuccessful.
 */
void *aml_area_mmap(struct aml_area *area, void *ptr, size_t size);
/* FIXME! */
int aml_area_available(const struct aml_area *area);
/*
 * Creates a new binding structure based on an existing Linux memory area.
 * "area": an initialized memory area structure.
 * "binding": an address where the pointer to the newly allocated binding
 *            structure will be stored.
 * Returns 0 if successful; an error code otherwise.
 */
int aml_area_binding(const struct aml_area *area, struct aml_binding **binding);

/*******************************************************************************
 * Tiling:
 * Representation of a data structure organization in memory.
 ******************************************************************************/

/* opaque handle to all tilings */
struct aml_tiling_data;
struct aml_tiling_iterator_data;

/*forward declarations */
struct aml_tiling_iterator_ops;
struct aml_tiling_iterator;


struct aml_tiling_ops {
	int (*create_iterator)(struct aml_tiling_data *tiling,
			       struct aml_tiling_iterator **iterator,
			       int flags);
	int (*init_iterator)(struct aml_tiling_data *tiling,
			     struct aml_tiling_iterator *iterator, int flags);
	int (*destroy_iterator)(struct aml_tiling_data *tiling,
				struct aml_tiling_iterator *iterator);
	int (*tileid)(const struct aml_tiling_data *tiling, va_list);
	size_t (*tilesize)(const struct aml_tiling_data *tiling, int tileid);
	void* (*tilestart)(const struct aml_tiling_data *tiling,
			   const void *ptr, int tileid);
	int (*ndims)(const struct aml_tiling_data *tiling, va_list);
};

struct aml_tiling {
	struct aml_tiling_ops *ops;
	struct aml_tiling_data *data;
};

/*
 * Provides the tile id of a tile.
 * "tiling": an initialized tiling structure.
 * Variadic arguments:
 *  - a list of size_t coordinates, one per dimension of the tiling.
 * Returns the id of the tile (that is, its order in memory), to use with other
 * functions.
 * Returns -1 in case of invalid coordinates.
 */
int aml_tiling_tileid(const struct aml_tiling *tiling, ...);

/*
 * Provides the information on the size of a tile.
 * "tiling": an initialized tiling structure.
 * "tileid": an identifier of a tile (a value between 0 and the number of tiles
 *           minus 1).
 * Returns the size of a tile.
 */
size_t aml_tiling_tilesize(const struct aml_tiling *tiling, int tileid);

/*
 * Provides the information on the location of a tile in memory.
 * "tiling": an initialized tiling structure.
 * "ptr": an address of the start of the complete user data structure that this
 *        tiling describes.
 * "tileid": an identifier of a tile (a value between 0 and the number of tiles
 *           minus 1).
 * Returns the address of the start of the tile identified by "tileid", within
 * the provided user data structure.
 */
void* aml_tiling_tilestart(const struct aml_tiling *tiling, const void *ptr,
			   int tileid);

/*
 * Provides the dimensions of the entire tiling in tiles.
 * "tiling": an initialized tiling structure.
 * Variadic arguments:
 *  - a list of (size_t *), one per dimension of the tiling.
 *  Will contain the size of each dimension in tiles upon return.
 * Returns 0 if successful, an error code otherwise.
 */
int aml_tiling_ndims(const struct aml_tiling *tiling, ...);

/*
 * Allocates and initializes a new tiling iterator.
 * "tiling": an initialized tiling structure.
 * "iterator": an address where the pointer to the newly allocated iterator
 *             structure will be stored.
 * "flags": reserved for future use; pass 0 for now.
 * Returns 0 if successful; an error code otherwise.
 */
int aml_tiling_create_iterator(struct aml_tiling *tiling,
			       struct aml_tiling_iterator **iterator,
			       int flags);
/*
 * Initializes a tiling iterator.
 * "tiling": an initialized tiling structure.
 * "iterator": an allocated tiling iterator structure.
 * "flags": reserved for future use; pass 0 for now.
 * Returns 0 if successful; an error code otherwise.
 */
int aml_tiling_init_iterator(struct aml_tiling *tiling,
			     struct aml_tiling_iterator *iterator, int flags);
/*
 * Tears down an initialized tiling iterator.
 * "tiling": an initialized tiling structure.
 * "iterator": an initialized tiling iterator structure.
 * Returns 0 if successful; an error code otherwise.
 */
int aml_tiling_destroy_iterator(struct aml_tiling *tiling,
				struct aml_tiling_iterator *iterator);

struct aml_tiling_iterator_ops {
	int (*reset)(struct aml_tiling_iterator_data *iterator);
	int (*next)(struct aml_tiling_iterator_data *iterator);
	int (*end)(const struct aml_tiling_iterator_data *iterator);
	int (*get)(const struct aml_tiling_iterator_data *iterator,
		   va_list args);
};

struct aml_tiling_iterator {
	struct aml_tiling_iterator_ops *ops;
	struct aml_tiling_iterator_data *data;
};

/*
 * Resets a tiling iterator to the first tile.
 * "iterator": an initialized tiling iterator structure.
 * Returns 0 if successful; an error code otherwise.
 */
int aml_tiling_iterator_reset(struct aml_tiling_iterator *iterator);
/*
 * Advances a tiling iterator to the next tile.
 * "iterator": an initialized tiling iterator structure.
 * Returns 0 if successful; an error code otherwise.
 */
int aml_tiling_iterator_next(struct aml_tiling_iterator *iterator);
/*
 * Checks whether the iterator is past the last tile.
 * "iterator": an initialized tiling iterator structure.
 * Returns 0 if the iterator points at a valid tile; 1 if it's past the last
 * tile.
 */
int aml_tiling_iterator_end(const struct aml_tiling_iterator *iterator);
/*
 * Queries the iterator.
 * "iterator": an initialized tiling iterator structure.
 * Variadic arguments:
 * - "x": an argument of type unsigned long*; on return gets filled with the
 *        identifier of the tile currently pointed to.
 * Returns 0 if successful; an error code otherwise.
 */
int aml_tiling_iterator_get(const struct aml_tiling_iterator *iterator, ...);

/* Tiling types passed to the tiling create()/init()/vinit() routines.  */
/* Regular, linear tiling with uniform tile sizes.  */
#define AML_TILING_TYPE_1D 0
#define AML_TILING_TYPE_2D_ROWMAJOR 1
#define AML_TILING_TYPE_2D_COLMAJOR 2

/*
 * Allocates and initializes a new tiling.
 * "tiling": an address where the pointer to the newly allocated tiling
 *           structure will be stored.
 * "type": see AML_TILING_TYPE_*.
 * Variadic arguments:
 * - if "type" equals AML_TILING_TYPE_1D, two additional arguments are needed:
 *   - "tilesize": an argument of type size_t; provides the size of each tile.
 *   - "totalsize": an argument of type size_t; provides the size of the
 *                  complete user data structure to be tiled.
 * - if "type" equals AML_TILING_TYPE_2D, four additional arguments are needed:
 *   - "tilesize": an argument of type size_t; provides the size of a tile.
 *   - "totalsize": an argument of type size_t; provides the size of the
 *                  complete user data structure to be tiled.
 *   - "rowsize": an argument of type size_t; the number of tiles in a row
 *   - "colsize": an argument of type size_t; the number of tiles in a column
 * Returns 0 if successful; an error code otherwise.
 */
int aml_tiling_create(struct aml_tiling **tiling, int type, ...);
/*
 * Initializes a tiling.  This is a varargs-variant of the aml_tiling_vinit()
 * routine.
 * "tiling": an allocated tiling structure.
 * "type": see aml_tiling_create().
 * Variadic arguments: see aml_tiling_create().
 * Returns 0 if successful; an error code otherwise.
 */
int aml_tiling_init(struct aml_tiling *tiling, int type, ...);
/*
 * Initializes a tiling.
 * "tiling": an allocated tiling structure.
 * "type": see aml_tiling_create().
 * "args": see the variadic arguments of aml_tiling_create().
 * Returns 0 if successful; an error code otherwise.
 */
int aml_tiling_vinit(struct aml_tiling *tiling, int type, va_list args);
/*
 * Tears down an initialized tiling.
 * "tiling": an initialized tiling structure.
 * "type": see aml_tiling_create().
 * Returns 0 if successful; an error code otherwise.
 */
int aml_tiling_destroy(struct aml_tiling *tiling, int type);

/*******************************************************************************
 * Tiling 1D:
 ******************************************************************************/

extern struct aml_tiling_ops aml_tiling_1d_ops;
extern struct aml_tiling_iterator_ops aml_tiling_iterator_1d_ops;

struct aml_tiling_1d_data {
	size_t blocksize;
	size_t totalsize;
};

struct aml_tiling_iterator_1d_data {
	size_t i;
	struct aml_tiling_1d_data *tiling;
};

#define AML_TILING_1D_DECL(name) \
	struct aml_tiling_1d_data __ ##name## _inner_data; \
	struct aml_tiling name = { \
		&aml_tiling_1d_ops, \
		(struct aml_tiling_data *)&__ ## name ## _inner_data, \
	};

#define AML_TILING_ITERATOR_1D_DECL(name) \
	struct aml_tiling_iterator_1d_data __ ##name## _inner_data; \
	struct aml_tiling_iterator name = { \
		&aml_tiling_iterator_1d_ops, \
		(struct aml_tiling_iterator_data *)&__ ## name ## _inner_data, \
	};

#define AML_TILING_1D_ALLOCSIZE (sizeof(struct aml_tiling_1d_data) + \
				 sizeof(struct aml_tiling))

#define AML_TILING_ITERATOR_1D_ALLOCSIZE \
	(sizeof(struct aml_tiling_iterator_1d_data) + \
	 sizeof(struct aml_tiling_iterator))

/*******************************************************************************
 * Tiling 2D:
 * a contiguous memory area composed of contiguous tiles arranged in 2D grid.
 ******************************************************************************/

extern struct aml_tiling_ops aml_tiling_2d_rowmajor_ops;
extern struct aml_tiling_ops aml_tiling_2d_colmajor_ops;
extern struct aml_tiling_iterator_ops aml_tiling_iterator_2d_ops;

struct aml_tiling_2d_data {
	size_t blocksize;
	size_t totalsize;
	size_t ndims[2]; /* # number of rows, # number of cols (in tiles) */
};

struct aml_tiling_iterator_2d_data {
	size_t i;
	struct aml_tiling_2d_data *tiling;
};

#define AML_TILING_2D_ROWMAJOR_DECL(name) \
	struct aml_tiling_2d_data __ ##name## _inner_data; \
	struct aml_tiling name = { \
		&aml_tiling_2d_rowmajor_ops, \
		(struct aml_tiling_data *)&__ ## name ## _inner_data, \
	};

#define AML_TILING_2D_COLMAJOR_DECL(name) \
	struct aml_tiling_2d_data __ ##name## _inner_data; \
	struct aml_tiling name = { \
		&aml_tiling_2d_colmajor_ops, \
		(struct aml_tiling_data *)&__ ## name ## _inner_data, \
	};

#define AML_TILING_ITERATOR_2D_DECL(name) \
	struct aml_tiling_iterator_2d_data __ ##name## _inner_data; \
	struct aml_tiling_iterator name = { \
		&aml_tiling_iterator_2d_ops, \
		(struct aml_tiling_iterator_data *)&__ ## name ## _inner_data, \
	};

#define AML_TILING_2D_ALLOCSIZE (sizeof(struct aml_tiling_2d_data) + \
				 sizeof(struct aml_tiling))

#define AML_TILING_ITERATOR_2D_ALLOCSIZE \
	(sizeof(struct aml_tiling_iterator_2d_data) + \
	 sizeof(struct aml_tiling_iterator))

/*******************************************************************************
 * Binding:
 * Representation of page bindings in an area
 ******************************************************************************/

/* opaque handle to all bindings */
struct aml_binding_data;

struct aml_binding_ops {
	int (*nbpages)(const struct aml_binding_data *binding,
		       const struct aml_tiling *tiling, const void *ptr,
		       int tileid);
	int (*pages)(const struct aml_binding_data *binding, void **pages,
		     const struct aml_tiling *tiling, const void *ptr,
		     int tileid);
	int (*nodes)(const struct aml_binding_data *binding, int *nodes,
		     const struct aml_tiling *tiling, const void *ptr,
		     int tileid);
};

struct aml_binding {
	struct aml_binding_ops *ops;
	struct aml_binding_data *data;
};

/*
 * Provides the size of a tile in memory, in pages.
 * "binding": an initialized binding structure.
 * "tiling": an initialized tiling structure.
 * "ptr", "tileid": see aml_tiling_tilestart().
 * Returns the total number of pages that a tile occupies, including partial
 * pages.
 */
int aml_binding_nbpages(const struct aml_binding *binding,
			const struct aml_tiling *tiling,
			const void *ptr, int tileid);
/*
 * Provides the addresses of pages that a tile occupies.
 * "binding": an initialized binding structure.
 * "pages": an array that will be filled with start addresses of all pages
 *          that a tile occupies.  The array must be at least
 *          aml_binding_nbpages() elements long.
 * "tiling": an initialized tiling structure.
 * "ptr", "tileid": see aml_tiling_tilestart().
 * Returns 0 if successful; an error code otherwise.
 */
int aml_binding_pages(const struct aml_binding *binding, void **pages,
		      const struct aml_tiling *tiling, const void *ptr,
		      int tileid);
/*
 * Provides the NUMA node information of pages that a tile occupies.
 * "binding": an initialized binding structure.
 * "nodes": an array that will be filled with NUMA node id's of all pages
 *          that a tile occupies.  The array must be at least
 *          aml_binding_nbpages() elements long.
 * "tiling": an initialized tiling structure.
 * "ptr", "tileid": see aml_tiling_tilestart().
 * Returns 0 if successful; an error code otherwise.
 */
int aml_binding_nodes(const struct aml_binding *binding, int *nodes,
		      const struct aml_tiling *tiling, const void *ptr,
		      int tileid);

/* Binding types passed to the binding create()/init()/vinit() routines.  */
/* Binding where all pages are bound to the same NUMA node.  */
#define AML_BINDING_TYPE_SINGLE 0
/* Binding where pages are interleaved among multiple NUMA nodes.  */
#define AML_BINDING_TYPE_INTERLEAVE 1

/*
 * Allocates and initializes a new binding.
 * "binding": an address where the pointer to the newly allocated binding
 *            structure will be stored.
 * "type": see AML_BINDING_TYPE_*.
 * Variadic arguments:
 * - if "type" equals AML_BINDING_TYPE_SINGLE, one additional argument is
 *   needed:
 *   - "node": an argument of type int; provides a NUMA node id where pages
 *             will be allocated from.
 * - if "type" equals AML_BINDING_TYPE_INTERLEAVE, one additional argument is
 *   needed:
 *   - "mask": an argument of type const struct aml_bitmap*; storing a bitmask of
 *             NUMA node ids where pages will be allocated from.  See
 *             aml_bitmap for more information.
 * Returns 0 if successful; an error code otherwise.
 */
int aml_binding_create(struct aml_binding **binding, int type, ...);
/*
 * Initializes a new binding.  This is a varags-variant of the
 * aml_binding_vinit() routine.
 * "binding": an allocated binding structure.
 * "type": see aml_binding_create().
 * Variadic arguments: see aml_binding_create().
 * Returns 0 if successful; an error code otherwise.
 */
int aml_binding_init(struct aml_binding *binding, int type, ...);
/*
 * Initializes a new binding.
 * "binding": an allocated binding structure.
 * "type": see aml_binding_create().
 * "args": see the variadic arguments of aml_binding_create().
 * Returns 0 if successful; an error code otherwise.
 */
int aml_binding_vinit(struct aml_binding *binding, int type, va_list args);
/*
 * Tears down an initialized binding.
 * "binding": an initialized binding structure.
 * "type": see aml_binding_create().
 * Returns 0 if successful; an error code otherwise.
 */
int aml_binding_destroy(struct aml_binding *binding, int type);

/*******************************************************************************
 * Single Binding:
 * All pages on the same node
 ******************************************************************************/

extern struct aml_binding_ops aml_binding_single_ops;

struct aml_binding_single_data {
	int node;
};

#define AML_BINDING_SINGLE_DECL(name) \
	struct aml_binding_single_data __ ##name## _inner_data; \
	struct aml_binding name = { \
		&aml_binding_single_ops, \
		(struct aml_binding_data *)&__ ## name ## _inner_data, \
	};

#define AML_BINDING_SINGLE_ALLOCSIZE (sizeof(struct aml_binding_single_data) + \
				      sizeof(struct aml_binding))

/*******************************************************************************
 * Interleave Binding:
 * each page, of each tile, interleaved across nodes.
 ******************************************************************************/

#define AML_MAX_NUMA_NODES AML_BITMAP_MAX

extern struct aml_binding_ops aml_binding_interleave_ops;

struct aml_binding_interleave_data {
	int nodes[AML_MAX_NUMA_NODES];
	int count;
};

#define AML_BINDING_INTERLEAVE_DECL(name) \
	struct aml_binding_interleave_data __ ##name## _inner_data; \
	struct aml_binding name = { \
		&aml_binding_interleave_ops, \
		(struct aml_binding_data *)&__ ## name ## _inner_data, \
	};

#define AML_BINDING_INTERLEAVE_ALLOCSIZE \
	(sizeof(struct aml_binding_interleave_data) + \
	 sizeof(struct aml_binding))

/*******************************************************************************
 * DMA:
 * Management of low-level movement of memory.
 ******************************************************************************/

/* Internal macros used for tracking DMA request types.  */
/* Invalid request type.  Used for marking inactive requests in the vector.  */
#define AML_DMA_REQUEST_TYPE_INVALID -1
/* Copy request type.  Uses memcpy() for data migration.  */
#define AML_DMA_REQUEST_TYPE_COPY 0
/* Move request type.  Uses move_pages() for data migration.  */
#define AML_DMA_REQUEST_TYPE_MOVE 1

struct aml_dma_request;
struct aml_dma_data;

struct aml_dma_ops {
	int (*create_request)(struct aml_dma_data *dma,
			      struct aml_dma_request **req, int type,
			      va_list args);
	int (*destroy_request)(struct aml_dma_data *dma,
			       struct aml_dma_request *req);
	int (*wait_request)(struct aml_dma_data *dma,
			    struct aml_dma_request *req);
};

struct aml_dma {
	struct aml_dma_ops *ops;
	struct aml_dma_data *data;
};

/*
 * Requests a synchronous data copy between two different tiles, using
 * memcpy() or equivalent.
 * "dma": an initialized DMA structure.
 * Variadic arguments:
 * - "dt": an argument of type struct aml_tiling*; the destination tiling
 *         structure.
 * - "dptr": an argument of type void*; the start address of the complete
 *           destination user data structure.
 * - "dtid": an argument of type int; the destination tile identifier.
 * - "st": an argument of type struct aml_tiling*; the source tiling structure.
 * - "sptr": an argument of type void*; the start address of the complete
 *           source user data structure.
 * - "stid": an argument of type int; the source tile identifier.
 * Returns 0 if successful; an error code otherwise.
 */
int aml_dma_copy(struct aml_dma *dma, ...);
/*
 * Requests a data copy between two different tiles.  This is an asynchronous
 * version of aml_dma_copy().
 * "dma": an initialized DMA structure.
 * "req": an address where the pointer to the newly assigned DMA request will be
 *        stored.
 * Variadic arguments: see aml_dma_copy().
 * Returns 0 if successful; an error code otherwise.
 */
int aml_dma_async_copy(struct aml_dma *dma, struct aml_dma_request **req, ...);
/*
 * Requests a synchronous data move of a tile to a new memory area, using
 * move_pages() or equivalent.
 * "dma": an initialized DMA structure.
 * Variadic arguments:
 * - "darea": an argument of type struct aml_area*; the destination memory area
 *         structure.
 * - "st": an argument of type struct aml_tiling*; the tiling structure.
 * - "sptr": an argument of type void*; the start address of the complete
 *           user data structure.
 * - "stid": an argument of type int; the tile identifier.
 * Returns 0 if successful; an error code otherwise.
 */
int aml_dma_move(struct aml_dma *dma, ...);
/*
 * Requests a data move of a tile to a new memory area.  This is an asynchronous
 * version of aml_dma_move().
 * "dma": an initialized DMA structure.
 * "req": an address where the pointer to the newly assigned DMA request will be
 *        stored.
 * Variadic arguments: see aml_dma_move().
 * Returns 0 if successful; an error code otherwise.
 *
 */
int aml_dma_async_move(struct aml_dma *dma, struct aml_dma_request **req, ...);
/*
 * Waits for an asynchronous DMA request to complete.
 * "dma": an initialized DMA structure.
 * "req": a DMA request obtained using aml_dma_async_*() calls.
 * Returns 0 if successful; an error code otherwise.
 */
int aml_dma_wait(struct aml_dma *dma, struct aml_dma_request *req);
/*
 * Tears down an asynchronous DMA request before it completes.
 * "dma": an initialized DMA structure.
 * "req": a DMA request obtained using aml_dma_async_*() calls.
 * Returns 0 if successful; an error code otherwise.
 */
int aml_dma_cancel(struct aml_dma *dma, struct aml_dma_request *req);

/*******************************************************************************
 * Scratchpad:
 * Use an area to stage data from another area in and out.
 * A dma handles the movement itself.
 ******************************************************************************/

struct aml_scratch_request;
struct aml_scratch_data;

/* Internal macros used for tracking scratchpad request types.  */
/* Invalid request type.  Used for marking inactive requests in the vector.  */
#define AML_SCRATCH_REQUEST_TYPE_INVALID -1
/* Push from the scratchpad to regular memory.  */
#define AML_SCRATCH_REQUEST_TYPE_PUSH 0
/* Pull from regular memory to the scratchpad.  */
#define AML_SCRATCH_REQUEST_TYPE_PULL 1
/* No-op/empty request */
#define AML_SCRATCH_REQUEST_TYPE_NOOP 2

struct aml_scratch_ops {
	int (*create_request)(struct aml_scratch_data *scratch,
			      struct aml_scratch_request **req, int type,
			      va_list args);
	int (*destroy_request)(struct aml_scratch_data *scratch,
			       struct aml_scratch_request *req);
	int (*wait_request)(struct aml_scratch_data *scratch,
			    struct aml_scratch_request *req);
	void *(*baseptr)(const struct aml_scratch_data *scratch);
	int (*release)(struct aml_scratch_data *scratch, int scratchid);
};

struct aml_scratch {
	struct aml_scratch_ops *ops;
	struct aml_scratch_data *data;
};

/*
 * Requests a synchronous pull from regular memory to the scratchpad.
 * "scratch": an initialized scratchpad structure.
 * Variadic arguments:
 * - "scratchptr": an argument of type void*; the scratchpad base pointer (see
 *                 aml_scratch_baseptr()).
 * - "scratchid": an argument of type int*; gets filled with the scratch tile
 *                identifier where the data will be pulled into.
 * - "srcptr": an argument of type void*; the start address of the complete
 *             source user data structure.
 * - "srcid": an argument of type int; the source tile identifier.
 * Returns 0 if successful; an error code otherwise.
 */
int aml_scratch_pull(struct aml_scratch *scratch, ...);
/*
 * Requests a pull from regular memory to the scratchpad.  This is an
 * asynchronous version of aml_scratch_pull().
 * "scratch": an initialized scratchpad structure.
 * "req": an address where the pointer to the newly assigned scratch request
 *        will be stored.
 * Variadic arguments: see aml_scratch_pull().
 * Returns 0 if successful; an error code otherwise.
 */
int aml_scratch_async_pull(struct aml_scratch *scratch,
			   struct aml_scratch_request **req, ...);
/*
 * Requests a synchronous push from the scratchpad to regular memory.
 * "scratch": an initialized scratchpad structure.
 * Variadic arguments:
 * - "dstptr": an argument of type void*; the start address of the complete
 *             destination user data structure.
 * - "dstid": an argument of type int*; gets filled with the destination tile
 *            identifier where the data will be pushed into (and where it was
 *            pulled from in the first place).
 * - "scratchptr": an argument of type void*; the scratchpad base pointer (see
 *                 aml_scratch_baseptr()).
 * - "scratchid": an argument of type int; the scratchpad tile identifier.
 * Returns 0 if successful; an error code otherwise.
 */
int aml_scratch_push(struct aml_scratch *scratch, ...);
/*
 * Requests a push from the scratchpad to regular memory.  This is an
 * asynchronous version of aml_scratch_push().
 * "scratch": an initialized scratchpad structure.
 * "req": an address where the pointer to the newly assigned scratch request
 *        will be stored.
 * Variadic arguments: see aml_scratch_push().
 * Returns 0 if successful; an error code otherwise.
 */
int aml_scratch_async_push(struct aml_scratch *scratch,
			   struct aml_scratch_request **req, ...);
/*
 * Waits for an asynchronous scratch request to complete.
 * "scratch": an initialized scratchpad structure.
 * "req": a scratch request obtained using aml_scratch_async_*() calls.
 * Returns 0 if successful; an error code otherwise.
 */
int aml_scratch_wait(struct aml_scratch *scratch,
		     struct aml_scratch_request *req);

/*
 * Tears down an asynchronous scratch request before it completes.
 * "scratch": an initialized scratchpad structure.
 * "req": a scratch request obtained using aml_scratch_async_*() calls.
 * Returns 0 if successful; an error code otherwise.
 */
int aml_scratch_cancel(struct aml_scratch *scratch,
		       struct aml_scratch_request *req);
/*
 * Provides the location of the scratchpad.
 * "scratch": an initialized scratchpad structure.
 * Returns a base pointer to the scratchpad memory buffer.
 */
void* aml_scratch_baseptr(const struct aml_scratch *scratch);

/*
 * Release a scratch tile for immediate reuse.
 * "scratch": an initialized scratchpad structure.
 * "scratchid": a scratchpad tile identifier.
 * Returns 0 if successful; an error code otherwise.
 */
int aml_scratch_release(struct aml_scratch *scratch, int scratchid);

/*******************************************************************************
 * General functions:
 * Initialize internal structures, cleanup everything at the end.
 ******************************************************************************/

/*
 * Initializes the library.
 * "argc": pointer to the main()'s argc argument; contents can get modified.
 * "argv": pointer to the main()'s argv argument; contents can get modified.
 * Returns 0 if successful; an error code otherwise.
 */
int aml_init(int *argc, char **argv[]);
/*
 * Terminates the library.
 * Returns 0 if successful; an error code otherwise.
 */
int aml_finalize(void);

#endif
