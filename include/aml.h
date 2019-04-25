/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
*******************************************************************************/

/**
 * \file aml.h
 *
 * \brief Main AML header file, contains all high level
 * abstractions declarations.
 **/

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

#include "aml/utils/bitmap.h"
#include "aml/utils/error.h"
#include "aml/utils/vector.h"
#include "aml/utils/version.h"

////////////////////////////////////////////////////////////////////////////////

/**
 * @}
 * @defgroup aml "AML Library functions"
 * @brief Initialize/Finalize library.
 *
 * General functions of aml library.
 * Initialization of internal structures, cleanup of everything at the end.
 *
 * @see aml_error
 * @see aml_version
 * @{
 **/

////////////////////////////////////////////////////////////////////////////////

/**
 * Initializes the library.
 * @param argc: pointer to the main()'s argc argument; contents can get
 *        modified.
 * @param argv: pointer to the main()'s argv argument; contents can get
 *        modified.
 * @return 0 if successful; an error code otherwise.
 **/
int aml_init(int *argc, char **argv[]);

/**
 * Terminates the library.
 * @return 0 if successful; an error code otherwise.
 **/
int aml_finalize(void);

////////////////////////////////////////////////////////////////////////////////

/**
 * @}
 * @defgroup aml_area "AML Area"
 * @brief Area High-Level API
 *
 * AML areas represent places where data can belong.
 * In shared memory systems, locality is a major concern for performance.
 * Beeing able to query memory from specific places is of a major interest
 * two achieve this goal. AML Areas provide mmap / munmap low level functions
 * to query memory from specific places materialized as areas. Available area
 * implementations dictate the way such places can be arranged and with which
 * properties. It is important to notice that function provided through Area API
 * are low-level functions and are not optimized for performance as allocator
 * are.
 *
 * @image html area.png "Illustration of areas on a copmlex system." width=700cm
 *
 * @see aml_area_linux
 *
 * @{
 **/

////////////////////////////////////////////////////////////////////////////////

/**
 * aml_area_data is an opaque handle defined by each aml_area
 * implementation. This not supposed to be used by end users.
 **/
struct aml_area_data;

/**
 * aml_area_ops is a structure containing implementations
 * of an area operations.
 * Aware users may create or modify implementation by assembling
 * appropriate operations in such a structure.
 **/
struct aml_area_ops {
	/**
	 * Building block for coarse grain allocator of virtual memory.
	 *
	 * @param data: Opaque handle to implementation specific data.
	 * @param ptr: A virtual address to be used by underlying
	 *        implementation.
	 *        Can be NULL.
	 * @param size: The minimum size of allocation.
	 *        Is greater than 0. Must not fail unless not enough
	 *        memory is available, or ptr argument does not point to a
	 *        suitable address.
	 *        In case of failure, aml_errno must be set to an appropriate
	 *        value.
	 * @return a pointer to allocated memory object.
	 **/
	void* (*mmap)(const struct aml_area_data  *data,
		      void                        *ptr,
		      size_t                       size);

	/**
	 * Building block for unmapping of virtual memory mapped with mmap()
	 * of the same area.
	 *
	 * @param data: An opaque handle to implementation specific data.
	 * @param ptr: Pointer to data mapped in physical memory. Cannot be
	 *        NULL.
	 * @param size: The size of data. Cannot be 0.
	 * @return: AML_AREA_* error code.
	 * @see mmap()
	 **/
	int (*munmap)(const struct aml_area_data *data,
		      void                       *ptr,
		      size_t                      size);
};

/**
 * An AML area is an implementation of memory operations for several type
 * of devices through a consistent abstraction.
 * This abstraction is meant to be implemented for several kind of devices,
 * i.e the same function calls allocate different kinds of devices depending
 * on the area implementation provided.
 **/
struct aml_area {
	/** Basic memory operations implementation **/
	struct aml_area_ops *ops;
	/** Implementation specific data. Set to NULL at creation. **/
	struct aml_area_data *data;
};

/**
 * Low-level function for getting memory from an area.
 * @param area: A valid area implementing access to target memory.
 * @param ptr: Implementation specific argument. See specific header.
 * @param size: The usable size of memory returned.
 * @return virtual memory from this area with at least queried size bytes.
 **/
void *aml_area_mmap(const struct aml_area *area,
		    void                 **ptr,
		    size_t                 size);

/**
 * Release data provided with aml_area_mmap() and the same area.
 * @param area: A valid area implementing access to target memory.
 * @param ptr: A pointer to memory address provided with aml_area_mmap()
 *        by same area and size.
 * @param size: The size of memory region pointed by "ptr".
 * @return an AML error code on operation success.
 * @see aml_area_mmap()
 **/
int
aml_area_munmap(const struct aml_area *area,
		void                  *ptr,
		size_t                 size);

////////////////////////////////////////////////////////////////////////////////

/**
 * @}
 * @defgroup aml_tiling "AML Tiling"
 * @brief Tiling Data Structure High-Level API
 *
 * Tiling is an array representation of data structures.
 * AML tiling structure can be defined as 1D or 2D contiguous arrays.
 * Tiles in tilings can be of custom size and AML provides iterators to
 * easily access tiles element.
 * @{
 **/

////////////////////////////////////////////////////////////////////////////////

/**
 * Tiling types passed to some tiling routines.
 * Regular, linear tiling with uniform tile sizes.
 **/
#define AML_TILING_TYPE_1D 0

/**
 * Tiling types passed to some tiling routines.
 * 2-dimensional cartesian tiling with uniform tile sizes, stored
 * in rowmajor order
 **/
#define AML_TILING_TYPE_2D_ROWMAJOR 1

/**
 * Tiling types passed to some tiling routines.
 * 2-dimensional cartesian tiling with uniform tile sizes, stored
 * in colmajor order
 **/
#define AML_TILING_TYPE_2D_COLMAJOR 2

/**
 * aml_tiling_data is an opaque handle defined by each aml_tiling
 * implementation. This not supposed to be used by end users.
 **/
struct aml_tiling_data;

/**
 * aml_area_tiling_iterator_data is an opaque handle defined by each
 * aml_tiling_iterator implementation. This not supposed to be used
 * by end users.
 **/
struct aml_tiling_iterator_data;

/**
 * aml_tiling_iterator_ops contains the specific operations defined
 * by an aml_tiling_iterator.
 * Aware users may create or modify implementation by assembling
 * appropriate operations in such a structure.
 **/
struct aml_tiling_iterator_ops;

/**
 * \brief aml_tiling_iterator is a structure for iterating over
 * elements of an aml_tiling.
 * \todo Provide a detailed explanation of what is a tiling iterator.
 **/
struct aml_tiling_iterator;

/**
 * aml_tiling_ops is a structure containing a set of operation
 * over a tiling. These operation are the creation and destruction
 * of iterators, access to tiles indexing, size and tiling dimension.
 * Aware users may create or modify implementation by assembling
 * appropriate operations in such a structure.
 **/
struct aml_tiling_ops {
	/**
	 * \todo Doc
	 **/
	int (*create_iterator)(struct aml_tiling_data *tiling,
			       struct aml_tiling_iterator **iterator,
			       int flags);
	/**
	 * \todo Doc
	 **/
	int (*init_iterator)(struct aml_tiling_data *tiling,
			     struct aml_tiling_iterator *iterator, int flags);
	/**
	 * \todo Doc
	 **/
	int (*fini_iterator)(struct aml_tiling_data *tiling,
				struct aml_tiling_iterator *iterator);
	/**
	 * \todo Doc
	 **/
	int (*destroy_iterator)(struct aml_tiling_data *tiling,
				struct aml_tiling_iterator **iterator);
	/**
	 * \todo Doc
	 **/
	int (*tileid)(const struct aml_tiling_data *tiling, va_list coords);
	/**
	 * \todo Doc
	 **/
	size_t (*tilesize)(const struct aml_tiling_data *tiling, int tileid);
	/**
	 * \todo Doc
	 **/
	void* (*tilestart)(const struct aml_tiling_data *tiling,
			   const void *ptr, int tileid);
	/**
	 * \todo Doc
	 **/
	int (*ndims)(const struct aml_tiling_data *tiling, va_list results);
};

/**
 * An aml_tiling is a multi-dimensional grid of data, e.g a matrix, a stencil
 * etc...
 * Tilings are used in AML as a description of a macro data structure that will
 * be used by a library for doing its own work. This structure is exploitable
 * by AML to perform optimized movement operations.
 **/
struct aml_tiling {
	/** @see aml_tiling_ops **/
	struct aml_tiling_ops *ops;
	/** @see aml_tiling_data **/
	struct aml_tiling_data *data;
};

/**
 Allocates and initializes a new tiling.
 * @param tiling: an address where the pointer to the newly allocated tiling
 *        structure will be stored.
 * @param type: see AML_TILING_TYPE_*.
 *        If "type" equals AML_TILING_TYPE_1D, two additional arguments are
 *        needed: tilesize, totalsize.
 *        If "type" equals AML_TILING_TYPE_2D, four additional arguments are
 *        needed: tilesize, totalsize, rowsize, colsize.
 * @param tilesize: an argument of type size_t; provides the size of a tile.
 * @param totalsize: an argument of type size_t; provides the size of the
 *        complete user data structure to be tiled.
 * @param rowsize: an argument of type size_t; the number of tiles in a row
 * @param colsize: an argument of type size_t; the number of tiles in a column
 * @return 0 if successful; an error code otherwise.
 **/
int aml_tiling_create(struct aml_tiling **tiling, int type, ...);

/**
 * Initializes a tiling.  This is a varargs-variant of the aml_tiling_vinit()
 * routine.
 * @param tiling: an allocated tiling structure.
 * @param type: see aml_tiling_create().
 * Variadic arguments: see aml_tiling_create().
 * @return 0 if successful; an error code otherwise.
 **/
int aml_tiling_init(struct aml_tiling *tiling, int type, ...);

/**
 * Initializes a tiling.
 * @param tiling: an allocated tiling structure.
 * @param type: see aml_tiling_create().
 * @param args: see the variadic arguments of aml_tiling_create().
 * @return 0 if successful; an error code otherwise.
 **/
int aml_tiling_vinit(struct aml_tiling *tiling, int type, va_list args);

/**
 * Tears down an initialized tiling.
 * @param tiling: an initialized tiling structure.
 * @param type: see aml_tiling_create().
 * @return 0 if successful; an error code otherwise.
 **/
int aml_tiling_destroy(struct aml_tiling *tiling, int type);

/**
 * Provides the tile id of a tile.
 * @param tiling: an initialized tiling structure.
 * @param coordinates: a list of size_t coordinates, one per dimension of the
 *        tiling.
 * @return -1 in case of invalid coordinates, else the id of the tile
 *         (that is, its order in memory), to use with other functions.
 **/
int aml_tiling_tileid(const struct aml_tiling *tiling, ...);

/**
 * Provides the information on the size of a tile.
 * @param tiling: an initialized tiling structure.
 * @param tileid: an identifier of a tile (a value between 0 and the number
 *        of tiles minus 1).
 * @return the size of a tile.
 **/
size_t aml_tiling_tilesize(const struct aml_tiling *tiling, int tileid);

/**
 * Provides the information on the location of a tile in memory.
 * @param tiling: an initialized tiling structure.
 * @param ptr: an address of the start of the complete user data structure
 *        that this tiling describes.
 * @param tileid: an identifier of a tile (a value between 0 and the number
 *        of tiles minus 1).
 * @return the address of the start of the tile identified by "tileid", within
 * the provided user data structure.
 **/
void *aml_tiling_tilestart(const struct aml_tiling *tiling,
			   const void *ptr,
			   int tileid);

/**
 * Provides the dimensions of the entire tiling in tiles.
 * @param tiling: an initialized tiling structure.
 * @param sizes: a list of output (size_t *), one per dimension of the tiling.
 *               Will contain the size of each dimension in tiles upon return.
 * @return 0 if successful, an error code otherwise.
 **/
int aml_tiling_ndims(const struct aml_tiling *tiling, ...);

/**
 * \todo Doc
 **/
struct aml_tiling_iterator_ops {
	/**
	 * \todo Doc
	 **/
	int (*reset)(struct aml_tiling_iterator_data *iterator);
	/**
	 * \todo Doc
	 **/
	int (*next)(struct aml_tiling_iterator_data *iterator);
	/**
	 * \todo Doc
	 **/
	int (*end)(const struct aml_tiling_iterator_data *iterator);
	/**
	 * \todo Doc
	 **/
	int (*get)(const struct aml_tiling_iterator_data *iterator,
		   va_list args);
};

/**
 * \todo Doc
 **/
struct aml_tiling_iterator {
	/** @see aml_tiling_iterator_ops **/
	struct aml_tiling_iterator_ops *ops;
	/** @see aml_tiling_iterator_data **/
	struct aml_tiling_iterator_data *data;
};

/**
 * Allocates and initializes a new tiling iterator.
 * @param tiling: an initialized tiling structure.
 * @param iterator: an address where the pointer to the newly allocated iterator
 *        structure will be stored.
 * @param flags: reserved for future use; pass 0 for now.
 * @return 0 if successful; an error code otherwise.
 **/
int aml_tiling_create_iterator(struct aml_tiling *tiling,
			       struct aml_tiling_iterator **iterator,
			       int flags);
/**
 * Initializes a tiling iterator.
 * @param tiling: an initialized tiling structure.
 * @param iterator: an allocated tiling iterator structure.
 * @param flags: reserved for future use; pass 0 for now.
 * @return 0 if successful; an error code otherwise.
 **/
int aml_tiling_init_iterator(struct aml_tiling *tiling,
			     struct aml_tiling_iterator *iterator, int flags);

/**
 * Finalize an initialized tiling iterator.
 * @param tiling: an initialized tiling structure.
 * @param iterator: an initialized tiling iterator structure.
 **/
void aml_tiling_fini_iterator(struct aml_tiling *tiling,
			      struct aml_tiling_iterator *iterator);
/**
 * Tears down an initialized tiling iterator.
 * @param tiling: an initialized tiling structure.
 * @param iterator: an initialized tiling iterator structure.
 * @return 0 if successful; an error code otherwise.
 **/
void aml_tiling_destroy_iterator(struct aml_tiling *tiling,
				struct aml_tiling_iterator **iterator);


/**
 * Resets a tiling iterator to the first tile.
 * @param iterator: an initialized tiling iterator structure.
 * @return 0 if successful; an error code otherwise.
 **/
int aml_tiling_iterator_reset(struct aml_tiling_iterator *iterator);

/**
 * Advances a tiling iterator to the next tile.
 * @param iterator: an initialized tiling iterator structure.
 * @return 0 if successful; an error code otherwise.
 **/
int aml_tiling_iterator_next(struct aml_tiling_iterator *iterator);

/**
 * Checks whether the iterator is past the last tile.
 * @param iterator: an initialized tiling iterator structure.
 * @return 0 if the iterator points at a valid tile; 1 if it's past the last
 * tile.
 **/
int aml_tiling_iterator_end(const struct aml_tiling_iterator *iterator);

/**
 * Queries the iterator.
 * @param iterator: an initialized tiling iterator structure.
 * @param x: an argument of type unsigned long*; on return gets filled with the
 *        identifier of the tile currently pointed to.
 * @return 0 if successful; an error code otherwise.
 **/
int aml_tiling_iterator_get(const struct aml_tiling_iterator *iterator, ...);

////////////////////////////////////////////////////////////////////////////////

/**
 * @}
 * @defgroup aml_dma "AML DMA"
 * @brief Management of low-level memory movements.
 *
 * AML DMA is the abstraction for handling memory movements.
 * AML DMA can asynchronously move data from one area to another.
 * While performing a movement, DMA operation
 * may also translates from a source tiling to a different
 * destination tiling.
 *
 * @image html dma.png width=600
 * @{
 **/

////////////////////////////////////////////////////////////////////////////////

/**
 * Internal macros used for tracking DMA request types.
 * Invalid request type.  Used for marking inactive requests in the vector.
 **/
#define AML_DMA_REQUEST_TYPE_INVALID -1

/**
 * Internal macros used for tracking DMA request types.
 * Copy request type.  Uses memcpy() for data migration.
 **/
#define AML_DMA_REQUEST_TYPE_COPY 0

/**
 * aml_dma is mainly used to asynchronously move data.
 * aml_dma_request is an opaque structure containing information
 * about ongoing request for data movement in a dma operation.
 * @see aml_dma_ops
 * @see aml_dma_async_copy()
 **/
struct aml_dma_request;

/**
 * Opaque handle implemented by each aml_dma implementations.
 * Should not be used by end-users.
 **/
struct aml_dma_data;

/**
 aml_dma_ops is a structure containing operations for a specific
 * aml_dma implementation.
 * These operation are operation are detailed in the structure.
 * They are specific in:
 * - the type of aml_area source and destination,
 * - the progress engine performing the operation,
 * - the type of of source and destination data structures.
 *
 * Each different combination of these three points may require a different
 * set of dma operations.
 **/
struct aml_dma_ops {
	/**
	 * Initiate a data movement, from a source pointer to a destination
	 * pointer, and output a request handler for managing the transfer.
	 * @param dma: dma_implementation internal data.
	 * @param req: Output the request handle to manage termination
	 *        of the movement.
	 * @param type: A valid AML_DMA_REQUEST_TYPE_* specifying the kind
	 *        of operation to perform.
	 * @param args: list of variadic arguments provided to aml_dma_copy()
	 * @return an AML error code.
	 **/
	int (*create_request)(struct aml_dma_data *dma,
			      struct aml_dma_request **req, int type,
			      va_list args);

	/**
	 * Destroy the request handle. If the data movement is still ongoing,
	 * then cancel it.
	 *
	 * @param dma: dma_implementation internal data.
	 * @param req: the request handle to manage termination of the movement.
	 * @return an AML error code.
	 **/
	int (*destroy_request)(struct aml_dma_data *dma,
			       struct aml_dma_request *req);

	/**
	 * Wait for termination of a data movement and destroy the request
	 * handle.
	 *
	 * @param dma: dma_implementation internal data.
	 * @param req: the request handle to manage termination of the movement.
	 * @return an AML error code.
	 **/
	int (*wait_request)(struct aml_dma_data *dma,
			    struct aml_dma_request *req);
};

/**
 * aml_dma is an abstraction for (asynchronously) moving data
 * from one area to another. The implementation of dma to use
 * is depends on the source and destination areas. The appropriate
 * dma choice is delegated to the user.
 * @see struct aml_area.
 **/
struct aml_dma {
	/** @see aml_dma_ops **/
	struct aml_dma_ops *ops;
	/** @see aml_dma_data **/
	struct aml_dma_data *data;
};

/**
 * Requests a synchronous data copy between two different tiles, using
 * memcpy() or equivalent.
 * @param dma: an initialized DMA structure.
 * @param dt: an argument of type struct aml_tiling*; the destination tiling
 *        structure.
 * @param dptr: an argument of type void*; the start address of the complete
 *        destination user data structure.
 * @param dtid: an argument of type int; the destination tile identifier.
 * @param st: an argument of type struct aml_tiling*; the source tiling
 *        structure.
 * @param sptr: an argument of type void*; the start address of the
 *        complete source user data structure.
 * @param stid: an argument of type int; the source tile identifier.
 * @return 0 if successful; an error code otherwise.
 **/
int aml_dma_copy(struct aml_dma *dma, ...);

/**
 * Requests a data copy between two different tiles.  This is an asynchronous
 * version of aml_dma_copy().
 * @param dma: an initialized DMA structure.
 * @param req: an address where the pointer to the newly assigned DMA request
 *        will be stored.
 * Variadic arguments: see aml_dma_copy().
 * @return 0 if successful; an error code otherwise.
 **/
int aml_dma_async_copy(struct aml_dma *dma, struct aml_dma_request **req, ...);

/**
 * Waits for an asynchronous DMA request to complete.
 * @param dma: an initialized DMA structure.
 * @param req: a DMA request obtained using aml_dma_async_*() calls.
 * @return 0 if successful; an error code otherwise.
 **/
int aml_dma_wait(struct aml_dma *dma, struct aml_dma_request *req);

/**
 * Tears down an asynchronous DMA request before it completes.
 * @param dma: an initialized DMA structure.
 * @param req: a DMA request obtained using aml_dma_async_*() calls.
 * @return 0 if successful; an error code otherwise.
 **/
int aml_dma_cancel(struct aml_dma *dma, struct aml_dma_request *req);

////////////////////////////////////////////////////////////////////////////////

/**
 * @}
 * @defgroup aml_scratch "AML Scratchpad"
 * @brief Stage-in, Stage-out High Level Abstraction.
 *
 * Scratchpad in an abstraction fro moving data back and forth from
 * a data representation in an area to another data representation in another
 * areas. This is especially usefull from moving to user data representation
 * to an architecure optimized representation for heavy computational work,
 * then returning the to user representation.
 * Data movement is performed with two dma engines from one area and tiling to
 * another area and tiling.
 *
 * @image html scratch.png width=600
 * @see aml_dma
 * @{
 **/

////////////////////////////////////////////////////////////////////////////////

/**
 * Scratch is mainly used to asynchronously move data back and forth between
 * two areas. aml_scratch_request is an opaque structure containing information
 * about ongoing requests for data movement.
 **/
struct aml_scratch_request;

/**
 * Opaque handle implemented by each scratches implementation.
 * Should not be used by end users.
 **/
struct aml_scratch_data;

/**
 * Scratchpad request types.
 * Invalid request type.  Used for marking inactive requests in the vector.
 **/
#define AML_SCRATCH_REQUEST_TYPE_INVALID -1

/**
 * Scratchpad request types.
 * Push from the scratchpad to regular memory.
 **/
#define AML_SCRATCH_REQUEST_TYPE_PUSH 0

/**
 * Scratchpad request types.
 * Pull from regular memory to the scratchpad.
 **/
#define AML_SCRATCH_REQUEST_TYPE_PULL 1

/**
 * Scratchpad request types.
 * No-op/empty request
 **/
#define AML_SCRATCH_REQUEST_TYPE_NOOP 2

/**
 * aml_scratch_ops contain a scratch implementation specific operations.
 * These operations implementation may vary depending on the source and
 * destination of data, and thus scratch implementations use different
 * operations.
 * Aware users may create or modify implementation by assembling
 * appropriate operations in such a structure.
 * @see struct aml_scratch
 **/
struct aml_scratch_ops {
	/**
	 * \todo Doc
	 **/
	int (*create_request)(struct aml_scratch_data *scratch,
			      struct aml_scratch_request **req, int type,
			      va_list args);
	/**
	 * \todo Doc
	 **/
	int (*destroy_request)(struct aml_scratch_data *scratch,
			       struct aml_scratch_request *req);
	/**
	 * \todo Doc
	 **/
	int (*wait_request)(struct aml_scratch_data *scratch,
			    struct aml_scratch_request *req);
	/**
	 * \todo Doc
	 **/
	void *(*baseptr)(const struct aml_scratch_data *scratch);
	/**
	 * \todo Doc
	 **/
	int (*release)(struct aml_scratch_data *scratch, int scratchid);
};

/**
 * An aml_scratch is abstraction aimed toward temporary use of a data structures
 * in a different area than the one where data currently resides. Scratches in
 * AML take care of asynchornously allocating and moving the data back and forth
 * between areas.
 **/
struct aml_scratch {
	/** @see aml_scratch_ops **/
	struct aml_scratch_ops *ops;
	/** @see aml_scratch_data **/
	struct aml_scratch_data *data;
};

/**
 * Requests a synchronous pull from regular memory to the scratchpad.
 * @param scratch: an initialized scratchpad structure.
 * @param scratchptr: an argument of type void*; the scratchpad base pointer.
 * @param scratchid: an argument of type int*; gets filled with the scratch tile
 *        identifier where the data will be pulled into.
 * @param srcptr: an argument of type void*; the start address of the complete
 *             source user data structure.
 * @param srcid: an argument of type int; the source tile identifier.
 * @see aml_scratch_baseptr()
 * @return 0 if successful; an error code otherwise.
 **/
int aml_scratch_pull(struct aml_scratch *scratch, ...);

/**
 * Requests a pull from regular memory to the scratchpad. This is an
 * asynchronous version of aml_scratch_pull().
 * @param scratch: an initialized scratchpad structure.
 * @param req: an address where the pointer to the newly assigned scratch
 *        request will be stored.
 * @param variadic arguments: see aml_scratch_pull().
 * @return 0 if successful; an error code otherwise.
 * @see aml_scratch_pull()
 **/
int aml_scratch_async_pull(struct aml_scratch *scratch,
			   struct aml_scratch_request **req, ...);
/**
 * Requests a synchronous push from the scratchpad to regular memory.
 * @param scratch: an initialized scratchpad structure.
 * @param dstptr: an argument of type void*; the start address of the complete
 *        destination user data structure.
 * @param dstid: an argument of type int*; gets filled with the destination tile
 *        identifier where the data will be pushed into (and where it was
 *        pulled from in the first place).
 * @param scratchptr: an argument of type void*; the scratchpad base pointer.
 * @param scratchid: an argument of type int; the scratchpad tile identifier.
 * @return 0 if successful; an error code otherwise.
 * @see aml_scratch_baseptr()
 **/
int aml_scratch_push(struct aml_scratch *scratch, ...);

/**
 * Requests a push from the scratchpad to regular memory.  This is an
 * asynchronous version of aml_scratch_push().
 * @param scratch: an initialized scratchpad structure.
 * @param req: an address where the pointer to the newly assigned scratch
 *        request will be stored.
 * Variadic arguments: see aml_scratch_push().
 * @return 0 if successful; an error code otherwise.
 * @see aml_scratch_push()
 **/
int aml_scratch_async_push(struct aml_scratch *scratch,
			   struct aml_scratch_request **req, ...);
/**
 * Waits for an asynchronous scratch request to complete.
 * @param scratch: an initialized scratchpad structure.
 * @param req: a scratch request obtained using aml_scratch_async_*() calls.
 * @return 0 if successful; an error code otherwise.
 **/
int aml_scratch_wait(struct aml_scratch *scratch,
		     struct aml_scratch_request *req);

/**
 * Tears down an asynchronous scratch request before it completes.
 * @param scratch: an initialized scratchpad structure.
 * @param req: a scratch request obtained using aml_scratch_async_*() calls.
 * @return 0 if successful; an error code otherwise.
 **/
int aml_scratch_cancel(struct aml_scratch *scratch,
		       struct aml_scratch_request *req);
/**
 * Provides the location of the scratchpad.
 * @param scratch: an initialized scratchpad structure.
 * @return a base pointer to the scratchpad memory buffer.
 **/
void *aml_scratch_baseptr(const struct aml_scratch *scratch);

/**
 * Release a scratch tile for immediate reuse.
 * @param scratch: an initialized scratchpad structure.
 * @param scratchid: a scratchpad tile identifier.
 * @return 0 if successful; an error code otherwise.
 **/
int aml_scratch_release(struct aml_scratch *scratch, int scratchid);

////////////////////////////////////////////////////////////////////////////////

/**
 * @}
 **/

#endif
