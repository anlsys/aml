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
#include <pthread.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

#include <aml/utils/version.h>
#include <aml/utils/bitmap.h>
#include <aml/utils/vector.h>
#include <aml/tiling/tiling.h>

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

/*******************************************************************************
 * Areas:
 * embeds information about a byte-addressable physical memory location as well
 * as binding policies over it.
 ******************************************************************************/

/* Error codes */
#define AML_AREA_SUCCESS  0 /* Function call succeeded */
#define AML_AREA_EINVAL  -1 /* Invalid argument provided */
#define AML_AREA_ENOTSUP -2 /* Function not implemented for this type of area */
#define AML_AREA_ENOMEM  -3 /* Allocation failed */
#define AML_AREA_EDOM    -4 /* One arguent is out of allowed bounds */

/**
 * An AML area is an implementation of memory operations for several type of devices
 * through a consistent abstraction and specialization of allocators 
 * with aml_local_area_create().
 *
 * This abstraction is meant to be implemented for several kind of devices,
 * i.e the same function calls allocate different kinds of devices depending
 * on the area implementation provided.
 *
 **/
struct aml_area;

/*******************
 * Linux area:
 *
 * "aml_bitmap": is a bitwise translation of linux struct bitmask. See <numa.h>
 * "flags": use one of aml_area_linux_flag_*. 
 *          See <numaif.h> for further explanations.
 *******************/

/* Implementation of process wide memory operations on host processor. */
extern struct aml_area *aml_area_linux_private;
/* Implementation of cross process memory operations on host processor. */
extern struct aml_area *aml_area_linux_shared;

/* Bind memory on given nodeset with MPOL_BIND policy */
const extern unsigned long aml_area_linux_flag_bind;
/* Bind memory on given nodeset with MPOL_INTERLEAVE policy */
const extern unsigned long aml_area_linux_flag_interleave;
/* Bind memory on given nodeset with MPOL_PREFFERED policy */
const extern unsigned long aml_area_linux_flag_preferred;

/*******************
 * hwloc area
 *
 * aml_bitmap is translated bitwise to hwloc_nodeset_t or hwloc_bitmap_t. 
 * See <aml/utils/hwloc.h> for conversion, <hwloc/bitmap.h>.
 * "flags" use flags in <aml/area/hwloc.h>. See <hwloc.h> for further explanations.
 *******************/

// Additional areas and area features in <aml/area/hwloc.h> if supported

/*******************
 * Global area features
 *******************/

/**
 * Create a specialized area from an existing area.
 * This is not a copy, callbacks are copied, the attributes are set to default.
 *
 * "area": The base area from which to create a new one. 
 * "binding": A bitmap to bind memory on subsequent memory mapping. Cannot be NULL.
 *            See specific areas doc in this header for setting bitmap.
 * "flags": flags associated with binding. (implementation dependent)
 *            See specific areas doc in this header for setting flags.
 * Returns NULL if area is NULL or does not support binding.
 **/
struct aml_area*
aml_local_area_create(struct aml_area         *area,
		      const struct aml_bitmap *binding,
		      const unsigned long      flags);

/**
 * Destroy specialized area.
 **/
void
aml_local_area_destroy(struct aml_area* area);

/**
 * Allocate memory for data. Allocation may be aligned
 * on a specific boundary if supported by area implementation. 
 * Allocations with aml_area_malloc() must be freed with aml_area_free().
 *
 * "area": The area implementation of malloc() function.
 * "ptr": A pointer where to store the allocation start address.
 * "size": The size to allocate.
 * "alignement": The data alignement. If 0, no alignement is performed.
 *               alignement may not be supported or support only specific values.
 *               
 * Returns AML_AREA_* error code.
 **/
int
aml_area_malloc(const struct aml_area *area,
		void                 **ptr,
		size_t                 size,
		size_t                 alignement);

/**
 * Release data allocated with aml_area_malloc() and the same area.
 * "area": The area implementation of free() function used for allocation.
 * "ptr": A pointer where to store the allocation start address.
 *
 * Returns AML_AREA_* error code.
 **/
int
aml_area_free(const struct aml_area *area,
	      void                  *ptr);

/*******************************************************************************
 * Tiling:
 * Representation of a data structure organization in memory.
 ******************************************************************************/

#include <aml/tiling/tiling.h>

struct aml_tiling;

/* Tiling types passed to the tiling create()/init()/vinit() routines.  */
/* Regular, linear tiling with uniform tile sizes.  */
#define AML_TILING_TYPE_1D 0
#define AML_TILING_TYPE_2D_ROWMAJOR 1
#define AML_TILING_TYPE_2D_COLMAJOR 2

/**
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
 ** Returns 0 if successful; an error code otherwise.
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

struct aml_tiling_iterator;

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

/*******************************************************************************
 * DMA:
 * Management of low-level movement of memory.
 ******************************************************************************/

struct aml_dma_request;
struct aml_dma_data;

struct aml_dma_ops {
	int (*create_request)(struct aml_dma_data *dma,
			      struct aml_dma_request **req,
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

#endif
