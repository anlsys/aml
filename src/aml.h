#ifndef AML_H
#define AML_H 1

#include <numa.h>
#include <memkind.h>
#include <stdlib.h>

/*******************************************************************************
 * Forward Declarations:
 ******************************************************************************/

struct aml_area;

/*******************************************************************************
 * Arenas:
 * In-memory allocator implementation. Dispatches actual memory mappings back to
 * areas.
 ******************************************************************************/

#define AML_ARENA_FLAG_ZERO 1

/* opaque handle to configuration data */
struct aml_arena_data;

struct aml_arena_ops {
	int (*create)(struct aml_arena_data *, struct aml_area *);
	int (*purge)(struct aml_arena_data *);
	void *(*mallocx)(struct aml_arena_data *, size_t, int);
	void (*dallocx)(struct aml_arena_data *, void *, int);
	void *(*reallocx)(struct aml_arena_data *, void *, size_t, int);
};

struct aml_arena {
	struct aml_arena_ops *ops;
	struct aml_arena_data *data;
};

int aml_arena_create(struct aml_arena *, struct aml_area *);
int aml_arena_purge(struct aml_arena *);
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

int aml_arena_jemalloc_regular_init(struct aml_arena_jemalloc_data *);
int aml_arena_jemalloc_regular_destroy(struct aml_arena_jemalloc_data *);
int aml_arena_jemalloc_aligned_init(struct aml_arena_jemalloc_data *, size_t);
int aml_arena_jemalloc_align_destroy(struct aml_arena_jemalloc_data *);
int aml_arena_jemalloc_generic_init(struct aml_arena_jemalloc_data *,
				    struct aml_arena_jemalloc_data *);
int aml_arena_jemalloc_generic_destroy(struct aml_arena_jemalloc_data *);

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

int aml_area_posix_init(struct aml_area_posix_data *);
int aml_area_posix_destroy(struct aml_area_posix_data *);

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

struct aml_area_linux_mbind_data {
	unsigned long nodemask[AML_NODEMASK_SZ];
	int policy;
};

struct aml_area_linux_mbind_ops {
	int (*pre_bind)(struct aml_area_linux_mbind_data *);
	int (*post_bind)(struct aml_area_linux_mbind_data *, void *, size_t);
};

int aml_area_linux_mbind_setdata(struct aml_area_linux_mbind_data *, int,
				 unsigned long *);
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

int aml_area_linux_init(struct aml_area_linux *);
int aml_area_linux_destroy(struct aml_area_linux *);

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
 * General functions:
 * Initialize internal structures, cleanup everything at the end.
 ******************************************************************************/

int aml_init(int *argc, char **argv[]);
int aml_finalize(void);

#endif
