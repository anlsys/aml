#ifndef AML_H
#define AML_H 1

#include <numa.h>
#include <memkind.h>
#include <stdlib.h>

/*******************************************************************************
 * Forward Declarations:
 ******************************************************************************/

struct aml_arena;
struct aml_area;
struct aml_dma;

/*******************************************************************************
 * Arenas:
 * interface between areas (user-facing low-level mapping) and actual memory
 * allocators (i.e. managers of buckets and so on).
 ******************************************************************************/

struct aml_arena {
	unsigned int uid;
	int (*init)(struct aml_arena *, struct aml_area*);
	int (*destroy)(struct aml_arena *);
	void *(*malloc)(struct aml_arena *, size_t);
	void (*free)(struct aml_arena *, void *);
	void *(*calloc)(struct aml_arena *, size_t, size_t);
	void *(*realloc)(struct aml_arena *, void *, size_t);
	void *(*acquire)(struct aml_arena *, size_t);
	void (*release)(struct aml_arena *, void *);
	void *extra;
};

/* jemalloc arena template */
extern struct aml_arena aml_arena_jemalloc;

int aml_arena_init(struct aml_arena *, struct aml_arena *, struct aml_area *);
int aml_arena_destroy(struct aml_arena *);

/*******************************************************************************
 * Areas:
 * embeds information about a byte-addressable physical memory location and well
 * as binding policies over it.
 ******************************************************************************/

struct aml_area {
	int (*init)(struct aml_area *);
	int (*destroy)(struct aml_area *);
	struct aml_arena* (*get_arena)(struct aml_area *);
	void * (*mmap)(struct aml_area *, void *, size_t);
	int (*mbind)(struct aml_area *, void *, size_t);
	int (*available)(struct aml_area *);
	void *extra;
};

/* templates for typical area types */
extern struct aml_area aml_area_hbm;
extern struct aml_area aml_area_regular;

int aml_area_init(struct aml_area *, struct aml_area *);
int aml_area_destroy(struct aml_area *);

/*******************************************************************************
 * Area allocations:
 * Low-level, direct allocation of memory from an area.
 ******************************************************************************/

void *aml_area_malloc(struct aml_area *, size_t);
void aml_area_free(struct aml_area *, void *);
void *aml_area_calloc(struct aml_area *, size_t, size_t);
void *aml_area_realloc(struct aml_area *, void *, size_t);
void *aml_area_acquire(struct aml_area *, size_t);
void aml_area_release(struct aml_area *, void *);

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
