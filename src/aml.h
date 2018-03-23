#ifndef AML_H
#define AML_H 1

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

int aml_arena_register(struct aml_arena *arena, struct aml_area *area);
int aml_arena_deregister(struct aml_arena *arena);
void *aml_arena_mallocx(struct aml_arena *arena, size_t size, int flags);
void aml_arena_dallocx(struct aml_arena *arena, void *ptr, int flags);
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


#define AML_ARENA_JEMALLOC_TYPE_REGULAR 0
#define AML_ARENA_JEMALLOC_TYPE_ALIGNED 1
#define AML_ARENA_JEMALLOC_TYPE_GENERIC 2

int aml_arena_jemalloc_create(struct aml_arena **arena, int type, ...);
int aml_arena_jemalloc_init(struct aml_arena *arena, int type, ...);
int aml_arena_jemalloc_vinit(struct aml_arena *arena, int type, va_list args);
int aml_arena_jemalloc_destroy(struct aml_arena *arena);

/*******************************************************************************
 * Areas:
 * embeds information about a byte-addressable physical memory location and well
 * as binding policies over it.
 ******************************************************************************/

/* opaque handle to configuration data */
struct aml_area_data;

struct aml_area_ops {
	void *(*malloc)(struct aml_area_data *area, size_t size);
	void (*free)(struct aml_area_data *area, void *ptr);
	void *(*calloc)(struct aml_area_data *area, size_t num, size_t size);
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

int aml_area_posix_create(struct aml_area **area);
int aml_area_posix_vinit(struct aml_area *area);
int aml_area_posix_init(struct aml_area *area);
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

int aml_area_linux_manager_single_init(struct aml_area_linux_manager_data *data,
				       struct aml_arena *arena);
int aml_area_linux_manager_single_destroy(struct aml_area_linux_manager_data *data);

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
	int (*pre_bind)(struct aml_area_linux_mbind_data *data);
	int (*post_bind)(struct aml_area_linux_mbind_data *data, void *ptr,
			 size_t size);
	int (*binding)(const struct aml_area_linux_mbind_data *data,
		       struct aml_binding **binding);
};

int aml_area_linux_mbind_setdata(struct aml_area_linux_mbind_data *data,
				 int policy, const unsigned long *nodemask);
int aml_area_linux_mbind_generic_binding(const struct aml_area_linux_mbind_data *data,
					 struct aml_binding **binding);
int aml_area_linux_mbind_regular_pre_bind(struct aml_area_linux_mbind_data *data);
int aml_area_linux_mbind_regular_post_bind(struct aml_area_linux_mbind_data *data,
					   void *ptr, size_t size);
int aml_area_linux_mbind_mempolicy_pre_bind(struct aml_area_linux_mbind_data *data);
int aml_area_linux_mbind_mempolicy_post_bind(struct aml_area_linux_mbind_data *data,
					   void *ptr, size_t size);
int aml_area_linux_mbind_init(struct aml_area_linux_mbind_data *data,
			      int policy, const unsigned long *nodemask);
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

void *aml_area_linux_mmap_generic(struct aml_area_linux_mmap_data *data,
				  void *ptr, size_t size);
int aml_area_linux_mmap_anonymous_init(struct aml_area_linux_mmap_data *data);
int aml_area_linux_mmap_fd_init(struct aml_area_linux_mmap_data *data, int fd,
				off_t offset);
int aml_area_linux_mmap_tmpfile_init(struct aml_area_linux_mmap_data *data,
				     char *template, size_t max);
int aml_area_linux_mmap_anonymous_destroy(struct aml_area_linux_mmap_data *data);
int aml_area_linux_mmap_fd_destroy(struct aml_area_linux_mmap_data *data);
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

#define AML_AREA_LINUX_MANAGER_TYPE_SINGLE 0

#define AML_AREA_LINUX_MBIND_TYPE_REGULAR 0
#define AML_AREA_LINUX_MBIND_TYPE_MEMPOLICY 1

#define AML_AREA_LINUX_MMAP_TYPE_ANONYMOUS 0
#define AML_AREA_LINUX_MMAP_TYPE_FD 1
#define AML_AREA_LINUX_MMAP_TYPE_TMPFILE 2

int aml_area_linux_create(struct aml_area **area, int manager_type,
			  int mbind_type, int mmap_type, ...);
int aml_area_linux_init(struct aml_area *area, int manager_type, int mbind_type,
			int mmap_type, ...);
int aml_area_linux_vinit(struct aml_area *area, int manager_type,
			 int mbind_type, int mmap_type, va_list args);
int aml_area_linux_destroy(struct aml_area *area);

/*******************************************************************************
 * Generic Area API:
 * Low-level, direct access to area logic.
 * For memory allocation function, follows the POSIX spec.
 ******************************************************************************/

void *aml_area_malloc(struct aml_area *area, size_t size);
void aml_area_free(struct aml_area *area, void *ptr);
void *aml_area_calloc(struct aml_area *area, size_t num, size_t size);
void *aml_area_realloc(struct aml_area *area, void *ptr, size_t size);
void *aml_area_acquire(struct aml_area *area, size_t size);
void aml_area_release(struct aml_area *area, void *ptr);
void *aml_area_mmap(struct aml_area *area, void *ptr, size_t size);
int aml_area_available(const struct aml_area *area);
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
	size_t (*tilesize)(const struct aml_tiling_data *tiling, int tileid);
	void* (*tilestart)(const struct aml_tiling_data *tiling,
			   const void *ptr, int tileid);
};

struct aml_tiling {
	struct aml_tiling_ops *ops;
	struct aml_tiling_data *data;
};

size_t aml_tiling_tilesize(const struct aml_tiling *tiling, int tileid);
void* aml_tiling_tilestart(const struct aml_tiling *tiling, const void *ptr,
			   int tileid);


int aml_tiling_create_iterator(struct aml_tiling *tiling,
			       struct aml_tiling_iterator **iterator,
			       int flags);
int aml_tiling_init_iterator(struct aml_tiling *tiling,
			     struct aml_tiling_iterator *iterator, int flags);
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

int aml_tiling_iterator_reset(struct aml_tiling_iterator *iterator);
int aml_tiling_iterator_next(struct aml_tiling_iterator *iterator);
int aml_tiling_iterator_end(const struct aml_tiling_iterator *iterator);
int aml_tiling_iterator_get(const struct aml_tiling_iterator *iterator, ...);

#define AML_TILING_TYPE_1D 0

int aml_tiling_create(struct aml_tiling **tiling, int type, ...);
int aml_tiling_init(struct aml_tiling *tiling, int type, ...);
int aml_tiling_vinit(struct aml_tiling *tiling, int type, va_list args);
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

int aml_binding_nbpages(const struct aml_binding *binding,
			const struct aml_tiling *tiling,
			const void *ptr, int tileid);
int aml_binding_pages(const struct aml_binding *binding, void **pages,
		      const struct aml_tiling *tiling, const void *ptr,
		      int tileid);
int aml_binding_nodes(const struct aml_binding *binding, int *nodes,
		      const struct aml_tiling *tiling, const void *ptr,
		      int tileid);

#define AML_BINDING_TYPE_SINGLE 0
#define AML_BINDING_TYPE_INTERLEAVE 1

int aml_binding_create(struct aml_binding **binding, int type, ...);
int aml_binding_init(struct aml_binding *binding, int type, ...);
int aml_binding_vinit(struct aml_binding *binding, int type, va_list args);
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

#define AML_DMA_REQUEST_TYPE_INVALID -1
#define AML_DMA_REQUEST_TYPE_COPY 0
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

int aml_dma_copy(struct aml_dma *dma, ...);
int aml_dma_async_copy(struct aml_dma *dma, struct aml_dma_request **req, ...);
int aml_dma_move(struct aml_dma *dma, ...);
int aml_dma_async_move(struct aml_dma *dma, struct aml_dma_request **req, ...);
int aml_dma_wait(struct aml_dma *dma, struct aml_dma_request *req);
int aml_dma_cancel(struct aml_dma *dma, struct aml_dma_request *req);

/*******************************************************************************
 * Linux Sequential DMA API:
 * DMA logic implemented based on general linux API, with the caller thread
 * used as the only execution thread.
 ******************************************************************************/

extern struct aml_dma_ops aml_dma_linux_seq_ops;

struct aml_dma_request_linux_seq {
	int type;
	void *dest;
	void *src;
	size_t size;
	int count;
	void **pages;
	int *nodes;
};

struct aml_dma_linux_seq_data {
	size_t size;
	struct aml_dma_request_linux_seq *requests;
};

struct aml_dma_linux_seq_ops {
	int (*do_copy)(struct aml_dma_linux_seq_data *dma,
		       struct aml_dma_request_linux_seq *req);
	int (*do_move)(struct aml_dma_linux_seq_data *dma,
		       struct aml_dma_request_linux_seq *req);
	int (*add_request)(struct aml_dma_linux_seq_data *dma,
			   struct aml_dma_request_linux_seq **req);
	int (*remove_request)(struct aml_dma_linux_seq_data *dma,
			      struct aml_dma_request_linux_seq **req);
};

struct aml_dma_linux_seq {
	struct aml_dma_linux_seq_ops ops;
	struct aml_dma_linux_seq_data data;
};

#define AML_DMA_LINUX_SEQ_DECL(name) \
	struct aml_dma_linux_seq __ ##name## _inner_data; \
	struct aml_dma name = { \
		&aml_dma_linux_seq_ops, \
		(struct aml_dma_data *)&__ ## name ## _inner_data, \
	};

#define AML_DMA_LINUX_SEQ_ALLOCSIZE \
	(sizeof(struct aml_dma_linux_seq) + \
	 sizeof(struct aml_dma))

int aml_dma_linux_seq_create(struct aml_dma **dma, ...);
int aml_dma_linux_seq_init(struct aml_dma *dma, ...);
int aml_dma_linux_seq_vinit(struct aml_dma *dma, va_list args);
int aml_dma_linux_seq_destroy(struct aml_dma *dma);

/*******************************************************************************
 * Linux Parallel DMA API:
 * DMA logic implemented based on general linux API, with the caller thread
 * used as the only execution thread.
 ******************************************************************************/

extern struct aml_dma_ops aml_dma_linux_par_ops;

struct aml_dma_linux_par_thread_data {
	int tid;
	pthread_t thread;
	struct aml_dma_linux_par *dma;
	struct aml_dma_request_linux_par *req;
};

struct aml_dma_request_linux_par {
	int type;
	void *dest;
	void *src;
	size_t size;
	int count;
	void **pages;
	int *nodes;
	struct aml_dma_linux_par_thread_data *thread_data;
};

struct aml_dma_linux_par_data {
	size_t nbrequests;
	size_t nbthreads;
	struct aml_dma_request_linux_par *requests;
};

struct aml_dma_linux_par_ops {
	void *(*do_thread)(void *);
	int (*do_copy)(struct aml_dma_linux_par_data *,
		       struct aml_dma_request_linux_par *, int tid);
	int (*do_move)(struct aml_dma_linux_par_data *,
		       struct aml_dma_request_linux_par *, int tid);
	int (*add_request)(struct aml_dma_linux_par_data *,
			   struct aml_dma_request_linux_par **);
	int (*remove_request)(struct aml_dma_linux_par_data *,
			      struct aml_dma_request_linux_par **);
};

struct aml_dma_linux_par {
	struct aml_dma_linux_par_ops ops;
	struct aml_dma_linux_par_data data;
};

#define AML_DMA_LINUX_PAR_DECL(name) \
	struct aml_dma_linux_par __ ##name## _inner_data; \
	struct aml_dma name = { \
		&aml_dma_linux_par_ops, \
		(struct aml_dma_data *)&__ ## name ## _inner_data, \
	};

#define AML_DMA_LINUX_PAR_ALLOCSIZE \
	(sizeof(struct aml_dma_linux_par) + \
	 sizeof(struct aml_dma))

int aml_dma_linux_par_create(struct aml_dma **, ...);
int aml_dma_linux_par_init(struct aml_dma *, ...);
int aml_dma_linux_par_vinit(struct aml_dma *, va_list);
int aml_dma_linux_par_destroy(struct aml_dma *);

/*******************************************************************************
 * Scratchpad:
 * Use an area to stage data from an another area in and out.
 * A dma handles the movement itself.
 ******************************************************************************/

struct aml_scratch_request;
struct aml_scratch_data;

#define AML_SCRATCH_REQUEST_TYPE_INVALID -1
#define AML_SCRATCH_REQUEST_TYPE_PUSH 0
#define AML_SCRATCH_REQUEST_TYPE_PULL 1

struct aml_scratch_ops {
	int (*create_request)(struct aml_scratch_data *scratch,
			      struct aml_scratch_request **req, int type,
			      va_list args);
	int (*destroy_request)(struct aml_scratch_data *scratch,
			       struct aml_scratch_request *req);
	int (*wait_request)(struct aml_scratch_data *scratch,
			    struct aml_scratch_request *req);
};

struct aml_scratch {
	struct aml_scratch_ops *ops;
	struct aml_scratch_data *data;
};

int aml_scratch_pull(struct aml_scratch *scratch, ...);
int aml_scratch_async_pull(struct aml_scratch *scratch,
			   struct aml_scratch_request **req, ...);
int aml_scratch_push(struct aml_scratch *scratch, ...);
int aml_scratch_async_push(struct aml_scratch *scratch,
			   struct aml_scratch_request **req, ...);
int aml_scratch_wait(struct aml_scratch *scratch,
		     struct aml_scratch_request *req);
int aml_scratch_cancel(struct aml_scratch *scratch,
		       struct aml_scratch_request *req);

/*******************************************************************************
 * Sequential scratchpad API:
 * Scratchpad uses calling thread to trigger dma movements.
 ******************************************************************************/

extern struct aml_scratch_ops aml_scratch_seq_ops;

struct aml_scratch_request_seq {
	int type;
	struct aml_tiling *stiling;
	void *srcptr;
	int srcid;
	struct aml_tiling *dtiling;
	void *dstptr;
	int dstid;
	struct aml_dma_request *dma_req;
};

struct aml_scratch_seq_data {
	struct aml_area *srcarea, *scratcharea;
	struct aml_tiling *srctiling, *scratchtiling;
	struct aml_dma *dma;
	size_t nbrequests;
	struct aml_scratch_request_seq *requests;
};

struct aml_scratch_seq_ops {
	int (*doit)(struct aml_scratch_seq_data *scratch,
		    struct aml_scratch_request_seq *req);
	int (*add_request)(struct aml_scratch_seq_data *scratch,
			   struct aml_scratch_request_seq **req);
	int (*remove_request)(struct aml_scratch_seq_data *scratch,
			      struct aml_scratch_request_seq **req);
};

struct aml_scratch_seq {
	struct aml_scratch_seq_ops ops;
	struct aml_scratch_seq_data data;
};

#define AML_SCRATCH_SEQ_DECL(name) \
	struct aml_scratch_seq __ ##name## _inner_data; \
	struct aml_scratch name = { \
		&aml_scratch_seq_ops, \
		(struct aml_scratch_data *)&__ ## name ## _inner_data, \
	};

#define AML_SCRATCH_SEQ_ALLOCSIZE \
	(sizeof(struct aml_scratch_seq) + \
	 sizeof(struct aml_scratch))

int aml_scratch_seq_create(struct aml_scratch **scratch, ...);
int aml_scratch_seq_init(struct aml_scratch *scratch, ...);
int aml_scratch_seq_vinit(struct aml_scratch *scratch, va_list args);
int aml_scratch_seq_destroy(struct aml_scratch *scratch);

/*******************************************************************************
 * General functions:
 * Initialize internal structures, cleanup everything at the end.
 ******************************************************************************/

int aml_init(int *argc, char **argv[]);
int aml_finalize(void);

#endif
