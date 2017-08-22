#ifndef AML_H
#define AML_H 1

#include <numa.h>
#include <memkind.h>
#include <stdlib.h>

/*******************************************************************************
 * Areas:
 * embeds information about a byte-addressable physical memory location and well
 * as binding policies over it.
 ******************************************************************************/

/* WARNING: kind must be the first argument for this library to work */
struct aml_area {
	memkind_t kind;
	struct bitmask *nodemask;
};

#define AML_AREA_TYPE_HBM 0
#define AML_AREA_TYPE_REGULAR 1
#define AML_AREA_TYPE_MAX 2

int aml_area_init(struct aml_area *, unsigned int type);
int aml_area_from_nodestring(struct aml_area *, unsigned int, const char *);
int aml_area_from_nodemask(struct aml_area *, unsigned int, struct bitmask *);
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
