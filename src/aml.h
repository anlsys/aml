#ifndef AML_H
#define AML_H 1

#include <inttypes.h>
#include <memkind.h>
#include <numa.h>
#include <numaif.h>
#include <pthread.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

#ifndef PAGE_SIZE
#define PAGE_SIZE 4096
#endif


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

#define AML_ARENA_FLAG_ZERO 1

/* opaque handle to configuration data */
struct aml_arena_data;

struct aml_arena_ops {
	int (*register_arena)(struct aml_arena_data *, struct aml_area *);
	int (*deregister_arena)(struct aml_arena_data *);
	void *(*mallocx)(struct aml_arena_data *, size_t, int);
	void (*dallocx)(struct aml_arena_data *, void *, int);
	void *(*reallocx)(struct aml_arena_data *, void *, size_t, int);
};

struct aml_arena {
	struct aml_arena_ops *ops;
	struct aml_arena_data *data;
};

int aml_arena_register(struct aml_arena *, struct aml_area *);
int aml_arena_deregister(struct aml_arena *);
void *aml_arena_mallocx(struct aml_arena *, size_t, int);
void aml_arena_dallocx(struct aml_arena *, void *, int);
void *aml_arena_reallocx(struct aml_arena *, void *, size_t, int);

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


#define AML_ARENA_JEMALLOC_TYPE_REGULAR 0
#define AML_ARENA_JEMALLOC_TYPE_ALIGNED 1
#define AML_ARENA_JEMALLOC_TYPE_GENERIC 2

int aml_arena_jemalloc_create(struct aml_arena **, int type, ...);
int aml_arena_jemalloc_init(struct aml_arena *, int type, ...);
int aml_arena_jemalloc_vinit(struct aml_arena *, int type, va_list);
int aml_arena_jemalloc_destroy(struct aml_arena *);

/*******************************************************************************
 * Areas:
 * embeds information about a byte-addressable physical memory location and well
 * as binding policies over it.
 ******************************************************************************/

/* opaque handle to configuration data */
struct aml_area_data;

struct aml_area_ops {
	void *(*malloc)(struct aml_area_data *, size_t);
	void (*free)(struct aml_area_data *, void *);
	void *(*calloc)(struct aml_area_data *, size_t, size_t);
	void *(*realloc)(struct aml_area_data *, void *, size_t);
	void *(*acquire)(struct aml_area_data *, size_t);
	void (*release)(struct aml_area_data *, void *);
	void *(*mmap)(struct aml_area_data *, void *ptr, size_t);
	int (*available)(struct aml_area_data *);
	int (*binding)(struct aml_area_data *, struct aml_binding **);
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

int aml_area_posix_create(struct aml_area **);
int aml_area_posix_vinit(struct aml_area *);
int aml_area_posix_init(struct aml_area *);
int aml_area_posix_destroy(struct aml_area *);

/*******************************************************************************
 * Linux Area:
 ******************************************************************************/

extern struct aml_area_ops aml_area_linux_ops;

struct aml_area_linux_manager_data {
	struct aml_arena *pool;
	size_t pool_size;
};

struct aml_area_linux_manager_ops {
	struct aml_arena *(*get_arena)(struct aml_area_linux_manager_data *data);
};

extern struct aml_area_linux_manager_ops aml_area_linux_manager_single_ops;

int aml_area_linux_manager_single_init(struct aml_area_linux_manager_data *,
				       struct aml_arena *);
int aml_area_linux_manager_single_destroy(struct aml_area_linux_manager_data *);

#define AML_MAX_NUMA_NODES 128
#define AML_NODEMASK_BYTES (AML_MAX_NUMA_NODES/8)
#define AML_NODEMASK_SZ (AML_NODEMASK_BYTES/sizeof(unsigned long))

#define AML_NODEMASK_NBITS (8*sizeof(unsigned long))
#define AML_NODEMASK_ELT(i) ((i) / AML_NODEMASK_NBITS)
#define AML_NODEMASK_BITMASK(i) ((unsigned long)1 << ((i) % AML_NODEMASK_NBITS))
#define AML_NODEMASK_ISSET(mask, i) \
	((mask[AML_NODEMASK_ELT(i)] & AML_NODEMASK_BITMASK(i)) != 0)
#define AML_NODEMASK_SET(mask, i) (mask[AML_NODEMASK_ELT(i)] |= AML_NODEMASK_BITMASK(i))
#define AML_NODEMASK_ZERO(mask) \
	do {								\
		for(unsigned int __i = 0; __i < AML_NODEMASK_SZ; __i++)	\
			mask[__i] = 0;					\
	} while(0)


struct aml_area_linux_mbind_data {
	unsigned long nodemask[AML_NODEMASK_SZ];
	int policy;
};

struct aml_area_linux_mbind_ops {
	int (*pre_bind)(struct aml_area_linux_mbind_data *);
	int (*post_bind)(struct aml_area_linux_mbind_data *, void *, size_t);
	int (*binding)(struct aml_area_linux_mbind_data *, struct aml_binding *);
};

int aml_area_linux_mbind_setdata(struct aml_area_linux_mbind_data *, int,
				 unsigned long *);
int aml_area_linux_mbind_generic_binding(struct aml_area_linux_mbind_data *,
					 struct aml_binding **);
int aml_area_linux_mbind_regular_pre_bind(struct aml_area_linux_mbind_data *);
int aml_area_linux_mbind_regular_post_bind(struct aml_area_linux_mbind_data *,
					   void *, size_t);
int aml_area_linux_mbind_mempolicy_pre_bind(struct aml_area_linux_mbind_data *);
int aml_area_linux_mbind_mempolicy_post_bind(struct aml_area_linux_mbind_data *,
					   void *, size_t);
int aml_area_linux_mbind_init(struct aml_area_linux_mbind_data *, int,
			      unsigned long *);
int aml_area_linux_mbind_destroy(struct aml_area_linux_mbind_data *);

extern struct aml_area_linux_mbind_ops aml_area_linux_mbind_regular_ops;
extern struct aml_area_linux_mbind_ops aml_area_linux_mbind_mempolicy_ops;

struct aml_area_linux_mmap_data {
	int prot;
	int flags;
	int fildes;
	off_t off;
};

struct aml_area_linux_mmap_ops {
	void *(*mmap)(struct aml_area_linux_mmap_data *, void *, size_t);
};

void *aml_area_linux_mmap_generic(struct aml_area_linux_mmap_data *, void *,
				  size_t);
int aml_area_linux_mmap_anonymous_init(struct aml_area_linux_mmap_data *);
int aml_area_linux_mmap_fd_init(struct aml_area_linux_mmap_data *, int, size_t);
int aml_area_linux_mmap_tmpfile_init(struct aml_area_linux_mmap_data *, char *,
				     size_t);
int aml_area_linux_mmap_anonymous_destroy(struct aml_area_linux_mmap_data *);
int aml_area_linux_mmap_fd_destroy(struct aml_area_linux_mmap_data *);
int aml_area_linux_mmap_tmpfile_destroy(struct aml_area_linux_mmap_data *);

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

#define AML_AREA_LINUX_MANAGER_TYPE_SINGLE 0

#define AML_AREA_LINUX_MBIND_TYPE_REGULAR 0
#define AML_AREA_LINUX_MBIND_TYPE_MEMPOLICY 1

#define AML_AREA_LINUX_MMAP_TYPE_ANONYMOUS 0
#define AML_AREA_LINUX_MMAP_TYPE_FD 1
#define AML_AREA_LINUX_MMAP_TYPE_TMPFILE 2

int aml_area_linux_create(struct aml_area **, int, int, int, ...);
int aml_area_linux_init(struct aml_area *, int, int, int, ...);
int aml_area_linux_vinit(struct aml_area *, int, int, int, va_list);
int aml_area_linux_destroy(struct aml_area *);

/*******************************************************************************
 * Generic Area API:
 * Low-level, direct access to area logic.
 * For memory allocation function, follows the POSIX spec.
 ******************************************************************************/

void *aml_area_malloc(struct aml_area *, size_t);
void aml_area_free(struct aml_area *, void *);
void *aml_area_calloc(struct aml_area *, size_t, size_t);
void *aml_area_realloc(struct aml_area *, void *, size_t);
void *aml_area_acquire(struct aml_area *, size_t);
void aml_area_release(struct aml_area *, void *);
void *aml_area_mmap(struct aml_area *, void *, size_t);
int aml_area_available(struct aml_area *);
int aml_area_binding(struct aml_area *, struct aml_binding **);

/*******************************************************************************
 * DMA Engines:
 * Low-level, direct movement of memory.
 * We haven't decided in our design how we want to deal with memcpy/move_pages
 * differences yet.
 ******************************************************************************/

struct aml_dma {
	int (*copy)(struct aml_dma *, void *, const void *, size_t);
	int (*move)(struct aml_dma *, struct aml_area *, struct aml_area *,
		    void *, size_t);
};

int aml_dma_init(struct aml_dma *, unsigned int);
int aml_dma_destroy(struct aml_dma *);
int aml_dma_copy(struct aml_dma *, void *, const void *, size_t);
int aml_dma_move(struct aml_dma *, struct aml_area *, struct aml_area *,
		 void *, size_t);


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
	int (*create_iterator)(struct aml_tiling_data *,
			       struct aml_tiling_iterator **, int);
	int (*init_iterator)(struct aml_tiling_data *,
			     struct aml_tiling_iterator *, int);
	int (*destroy_iterator)(struct aml_tiling_data *,
				struct aml_tiling_iterator *);
	size_t (*tilesize)(struct aml_tiling_data *, va_list);
	void* (*tilestart)(struct aml_tiling_data *, void *, va_list);
};

struct aml_tiling {
	struct aml_tiling_ops *ops;
	struct aml_tiling_data *data;
};

size_t aml_tiling_tilesize(struct aml_tiling *, ...);
size_t aml_tiling_vtilesize(struct aml_tiling *, va_list);
void* aml_tiling_tilestart(struct aml_tiling *, void *, ...);
void* aml_tiling_vtilestart(struct aml_tiling *, void *, va_list);


int aml_tiling_create_iterator(struct aml_tiling *,
			       struct aml_tiling_iterator **, int);
int aml_tiling_init_iterator(struct aml_tiling *,
			     struct aml_tiling_iterator *, int);
int aml_tiling_destroy_iterator(struct aml_tiling *,
				struct aml_tiling_iterator *);

struct aml_tiling_iterator_ops {
	int (*reset)(struct aml_tiling_iterator_data *);
	int (*next)(struct aml_tiling_iterator_data *);
	int (*end)(struct aml_tiling_iterator_data *);
	int (*get)(struct aml_tiling_iterator_data *, va_list);
};

struct aml_tiling_iterator {
	struct aml_tiling_iterator_ops *ops;
	struct aml_tiling_iterator_data *data;
};

int aml_tiling_iterator_reset(struct aml_tiling_iterator *);
int aml_tiling_iterator_next(struct aml_tiling_iterator *);
int aml_tiling_iterator_end(struct aml_tiling_iterator *);
int aml_tiling_iterator_get(struct aml_tiling_iterator *, ...);

#define AML_TILING_TYPE_1D 0

int aml_tiling_create(struct aml_tiling **, int type, ...);
int aml_tiling_init(struct aml_tiling *, int type, ...);
int aml_tiling_vinit(struct aml_tiling *, int type, va_list);
int aml_tiling_destroy(struct aml_tiling *, int type);

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
 * Binding:
 * Representation of page bindings in an area
 ******************************************************************************/

/* opaque handle to all bindings */
struct aml_binding_data;

struct aml_binding_ops {
	int (*nbpages)(struct aml_binding_data *, struct aml_tiling *,
		       void *, va_list);
	int (*pages)(struct aml_binding_data *, void **, struct aml_tiling *,
		     void *, va_list);
	int (*nodes)(struct aml_binding_data *, int *, struct aml_tiling *,
		     void *, va_list);
};

struct aml_binding {
	struct aml_binding_ops *ops;
	struct aml_binding_data *data;
};

int aml_binding_nbpages(struct aml_binding *, struct aml_tiling *, void*, ...);
int aml_binding_pages(struct aml_binding *, void **, struct aml_tiling *, void*, ...);
int aml_binding_nodes(struct aml_binding *, int *, struct aml_tiling *, void *, ...);

#define AML_BINDING_TYPE_SINGLE 0
#define AML_BINDING_TYPE_INTERLEAVE 1

int aml_binding_create(struct aml_binding **, int type, ...);
int aml_binding_init(struct aml_binding *, int type, ...);
int aml_binding_vinit(struct aml_binding *, int type, va_list);
int aml_binding_destroy(struct aml_binding *, int type);

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
 * General functions:
 * Initialize internal structures, cleanup everything at the end.
 ******************************************************************************/

int aml_init(int *argc, char **argv[]);
int aml_finalize(void);

#endif
