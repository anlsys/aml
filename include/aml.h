/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

/**
 * \file aml.h
 *
 * \brief Main AML header file, contains all high level
 * abstractions declarations.
 **/

#ifndef AML_H
#define AML_H 1

#include <assert.h>
#include <errno.h>
#include <inttypes.h>
#include <numa.h>
#include <numaif.h>
#include <pthread.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/mman.h>
#include <unistd.h>

#include "aml/utils/bitmap.h"
#include "aml/utils/error.h"
#include "aml/utils/inner-malloc.h"
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
 * @defgroup aml_layout "AML Layout"
 * @brief Low level description of data orrganization at the byte granularity.
 *
 * Layout describes how contiguous element of a flat memory address space are
 * organized into a multidimensional array of elements of a fixed size.
 * The abstraction provide functions to build layouts, access elements,
 * reshape a layout, or subset a layout.
 *
 * Layouts are characterized by:
 * * A pointer to the data it describes
 * * A set of dimensions on which data spans.
 * * A stride in between elements of a dimension.
 * * A pitch indicating the space between contiguous elements of a dimension.
 *
 * The figure below describes a 2D layout with a sub-layout
 * (obtained with aml_layout_slice()) operation. The sub-layout has a stride
 * of 1 element along the second dimension. The slice has an offset of 1 element
 * along the same dimension, and its pitch is the pitch of the original
 * layout. Calling aml_layout_deref() on this sublayout with appropriate
 * coordinates will return a pointer to elements noted (coor_x, coord_y).
 * @see aml_layout_slice()
 *
 * @image html layout.png "2D layout with a 2D slice." width=400cm
 *
 * Access to specific elements of a layout can be done with
 * the aml_layout_deref() function. Access to an element is always done
 * relatively to the dimensions order set by at creation time.
 * However, internally, the library will store dimensions from the last
 * dimension to the first dimension such that elements along the first dimension
 * are contiguous in memory. This order is defined called with the value
 * AML_LAYOUT_ORDER_FORTRAN. Therefore, AML provides access to elements
 * without the overhead of user order choice through function suffixed
 * with "native".
 * @see aml_layout_deref()
 * @see aml_layout_deref_native()
 * @see aml_layout_dims_native()
 * @see aml_layout_slice_native()
 *
 * The layout abstraction also provides a function to reshape data
 * with a different set of dimensions. A reshaped layout will access
 * the same data but with different coordinates as pictured in the
 * figure below.
 * @see aml_layout_reshape()
 *
 * @image html reshape.png "2D layout turned into a 3D layout." width=700cm
 *
 * @see aml_layout_dense
 * @see aml_layout_pad
 * @{
 **/

////////////////////////////////////////////////////////////////////////////////

struct aml_layout_ops;
struct aml_layout_data;

/** Structure definition of AML layouts **/
struct aml_layout {
	/** Layout functions implementation **/
	struct aml_layout_ops *ops;
	/** Implementation specific data of a layout**/
	struct aml_layout_data *data;
};

/** List of operators implemented by layouts. **/
struct aml_layout_ops {
	/**
	 * Layout must provide a way to access a specific element
	 * according to the provided dimensions.
	 * Coordinates bounds checking is done in the generic API.
	 * Coordinates provided by the user will match the order
	 * Of the dimensions provided by the user in the constructor.
	 * However, dimensions are always stored internally in the
	 * AML_LAYOUT_ORDER_FORTRAN order.
	 * @param data[in]: The non-NULL handle to layout internal data.
	 * @param coords[in]: The non-NULL coordinates on which to access data.
	 * Coordinates are checked to be valid in aml_layout_deref().
	 * @return A pointer to the dereferenced element on success.
	 * @return NULL on failure with aml_errno set to the error reason.
	 **/
	void *(*deref)(const struct aml_layout_data *data,
		       const size_t *coords);

	/**
	 * Function for derefencing elements of a layout inside the library.
	 * Layout assumes data is always stored in AML_LAYOUT_ORDER_FORTRAN
	 * order. Coordinates provided by the library will match the same
	 * order, i.e last dimension first.
	 * @param data[in]: The non-NULL handle to layout internal data.
	 * @param coords[in]: The non-NULL coordinates on which to access data.
	 * The first coordinate should be the last dimensions and so on to the
	 * last, coordinate, last dimension.
	 * @return A pointer to the dereferenced element on success.
	 * @return NULL on failure with aml_errno set to the error reason.
	 **/
	void *(*deref_native)(const struct aml_layout_data *data,
			      const size_t *coords);

	/**
	 * Get the order in which dimensions of the layout are
	 * supposed to be accessed by the user.
	 * @param data[in]: The non-NULL handle to layout internal data.
	 * @return Order value. It is a bitmask with order bit set (or not set).
	 * Output value can be further checked against order AML_LAYOUT_ORDER
	 * flags by using the macro AML_LAYOUT_ORDER() on output value.
	 * @see AML_LAYOUT_ORDER()
	 **/
	int (*order)(const struct aml_layout_data *data);

	/**
	 * Return the layout dimensions in the user order.
	 * @param data[in]: The non-NULL handle to layout internal data.
	 * @param dims[out]: The non-NULL array of dimensions to fill. It is
	 * supposed to be large enough to contain ndims() elements.
	 * @return AML_SUCCESS on success, else an AML error code.
	 **/
	int (*dims)(const struct aml_layout_data *data, size_t *dims);

	/**
	 * Return the layout dimensions in the order they are actually stored
	 * in the library.
	 * @param data[in]: The non-NULL handle to layout internal data.
	 * @param dims[out]: The non-NULL array of dimensions to fill. It is
	 * supposed to be large enough to contain ndims() elements.
	 * @return AML_SUCCESS on success, else an AML error code.
	 **/
	int (*dims_native)(const struct aml_layout_data *data,
			   size_t *dims);

	/**
	 * Return the number of dimensions in a layout.
	 * @param data[in]: The non-NULL handle to layout internal data.
	 * @return The number of dimensions in the layout.
	 **/
	size_t (*ndims)(const struct aml_layout_data *data);

	/**
	 * Return the size of layout elements.
	 * @param data[in]: The non-NULL handle to layout internal data.
	 * @return The size of elements stored with this layout.
	 **/
	size_t (*element_size)(const struct aml_layout_data *data);

	/**
	 * Reshape the layout with different dimensions.
	 * Layout dimensions are checked in aml_layout_reshape() to store
	 * the exact same number of elements.
	 * @param data[in]: The non-NULL handle to layout internal data.
	 * @param output[out]: A non NULL pointer to a layout where to allocate
	 * a new layout resulting from the reshape operation.
	 * @param ndims[in]: The number of dimensions of the new layout.
	 * @param dims[in]: The number of elements along each dimension of
	 * the new layout.
	 * @return AML_SUCCESS on success, else an AML error code (<0).
	 **/
	int (*reshape)(const struct aml_layout_data *data,
		       struct aml_layout **output,
		       const size_t ndims,
		       const size_t *dims);

	/**
	 * Return a layout that is a subset of another layout.
	 * Slice arguments compatibility with the original layout are
	 * checked in aml_layout_slice().
	 * @param data[in]: The non-NULL handle to layout internal data.
	 * @param output[out]: A non NULL pointer to a layout where to allocate
	 * a new layout resulting from the slice operation.
	 * @param dims[in]: The number of elements of the slice along each
	 * dimension .
	 * @param offsets[in]: The index of the first element of the slice
	 * in each dimension.
	 * @param strides[in]: The displacement (in number of elements) between
	 * elements of the slice.
	 * @return A newly allocated layout with the queried subset of the
	 * original layout on succes.
	 * @return NULL on error with aml_errno set to the failure reason.
	 **/
	int (*slice)(const struct aml_layout_data *data,
		     struct aml_layout **output,
		     const size_t *dims,
		     const size_t *offsets,
		     const size_t *strides);

	/**
	 * Return a layout that is a subset of another layout, assuming
	 * dimensions are stored with AML_LAYOUT_ORDER_FORTRAN.
	 * Slice arguments compatibility with the original layout are
	 * checked in aml_layout_slice().
	 * @param data[in]: The non-NULL handle to layout internal data.
	 * @param output[out]: A non NULL pointer to a layout where to allocate
	 * a new layout resulting from the slice operation.
	 * @param dims[in]: The number of elements of the slice along each
	 * dimension .
	 * @param offsets[in]: The index of the first element of the slice
	 * in each dimension.
	 * @param strides[in]: The displacement (in number of elements) between
	 * elements of the slice.
	 * @return A newly allocated layout with the queried subset of the
	 * original layout on succes.
	 * @return NULL on error with aml_errno set to the failure reason.
	 **/
	int (*slice_native)(const struct aml_layout_data *data,
			    struct aml_layout **output,
			    const size_t *dims,
			    const size_t *offsets,
			    const size_t *strides);
};

/**
 * Tag specifying user storage of dimensions inside a layout.
 * Layout order is the first bit in an integer bitmask.
 * @see AML_LAYOUT_ORDER()
 * This tag will store dimensions in the order provided by the user,
 * i.e elements of the last dimension will be contiguous in memory.
 **/
#define AML_LAYOUT_ORDER_C (0<<0)

/**
 * Tag specifying user storage of dimensions inside a layout.
 * Layout order is the first bit in an integer bitmask.
 * @see AML_LAYOUT_ORDER()
 * This tag will store dimensions in the reversed order provided
 * by the user, i.e elements of the first dimension will be contiguous
 * in memory. This storage is the actual storage used by the library
 * inside the structure.
 **/
#define AML_LAYOUT_ORDER_FORTRAN (1<<0)

/**
 * This is equivalent to AML_LAYOUT_ORDER_C.
 * @see AML_LAYOUT_ORDER_C
 **/
#define AML_LAYOUT_ORDER_COLUMN_MAJOR (0<<0)

/**
 * This is equivalent to AML_LAYOUT_ORDER_FORTRAN.
 * @see AML_LAYOUT_ORDER_FORTRAN
 **/
#define AML_LAYOUT_ORDER_ROW_MAJOR (1<<0)

/**
 * Get the order bit of an integer bitmask.
 * The value can be further checked for equality
 * with AML_LAYOUT_ORDER_* values.
 * @param x: An integer with the first bit set
 * to the order value.
 * @return An integer containing only the bit order.
 **/
#define AML_LAYOUT_ORDER(x) ((x) & (1<<0))

/**
 * Dereference an element of a layout by its coordinates.
 * @param layout[in]: An initialized layout.
 * @param coords[in]: The coordinates on which to access data.
 * @return A pointer to the dereferenced element on success.
 * @return NULL on failure with aml_errno set to the error reason:
 * * AML_EINVAL if coordinate are out of bound
 * * See specific implementation of layout for further information
 * on possible error codes.
 **/
void *aml_layout_deref(const struct aml_layout *layout,
		       const size_t *coords);

/**
 * Equivalent to aml_layout_deref() but with bound checking
 * on coordinates.
 * @see aml_layout_deref()
 **/
void *aml_layout_deref_safe(const struct aml_layout *layout,
			    const size_t *coords);

/**
 * Get the order in which dimensions of the layout are supposed to be
 * accessed by the user.
 * @param layout[in]: An initialized layout.
 * @return The order (>0) on success, an AML error (<0) on failure.
 * @return On success, a bitmask with order bit set (or not set).
 * Output value can be further checked against order AML_LAYOUT_ORDER
 * flags by using the macro AML_LAYOUT_ORDER() on output value.
 * @see AML_LAYOUT_ORDER()
 **/
int aml_layout_order(const struct aml_layout *layout);

/**
 * Return the layout dimensions in the user order.
 * @param layout[in]: An initialized layout.
 * @param dims[out]: The non-NULL array of dimensions to fill. It is
 * supposed to be large enough to contain ndims() elements.
 * @return AML_SUCCESS on success, else an AML error code.
 **/
int aml_layout_dims(const struct aml_layout *layout, size_t *dims);

/**
 * Return the number of dimensions in a layout.
 * @param layout[in]: An initialized layout.
 * @return The number of dimensions in the layout.
 **/
size_t aml_layout_ndims(const struct aml_layout *layout);

/**
 * @brief Return the size of layout elements.
 * @param layout[in]: An initialized layout.
 * @return The size of elements stored with this layout.
 **/
size_t aml_layout_element_size(const struct aml_layout *layout);

/**
 * @brief Reshape the layout with different dimensions.
 * This function checks that the number of elements of
 * the reshaped layout matches the number of elements
 * in the original layout. Additional constraint may apply
 * depending on the layout implementation.
 * @param layout[in]: An initialized layout.
 * @param reshaped_layout[out]: A newly allocated layout
 * with the queried shape on succes.
 * @param ndims[in]: The number of dimensions of the new layout.
 * @param dims[in]: The number of elements along each dimension of
 * the new layout.
 * @return AML_SUCCESS on success.
 * @return AML_EINVAL if reshape dimensions are not compatible
 * with original layout dimensions.
 * @return AML_ENOMEM if AML failed to allocate the new structure.
 * @return Another aml_error code. Refer to the layout
 * implementation of reshape function.
 **/
int aml_layout_reshape(const struct aml_layout *layout,
		       struct aml_layout **reshaped_layout,
		       const size_t ndims,
		       const size_t *dims);

/**
 * Return a layout that is a subset of another layout.
 * The number of elements to subset along each dimension
 * must be compatible with offsets and strides.
 * This function checks that the amount of elements along
 * each dimensions of the slice actually fits in the original
 * layout.
 * @param layout[in]: An initialized layout.
 * @param reshaped_layout[out]: a pointer where to store a
 * newly allocated layout with the queried subset of the
 * original layout on succes.
 * @param dims[in]: The number of elements of the slice along each
 * dimension .
 * @param offsets[in]: The index of the first element of the slice
 * in each dimension. If NULL, offset is set to 0.
 * @param strides[in]: The displacement (in number of elements) between
 * elements of the slice. If NULL, stride is set to 1.
 * @return AML_SUCCESS on success, else an AML error code (<0).
 **/
int aml_layout_slice(const struct aml_layout *layout,
		     struct aml_layout **reshaped_layout,
		     const size_t *dims,
		     const size_t *offsets,
		     const size_t *strides);

////////////////////////////////////////////////////////////////////////////////

/**
 * @}
 * @defgroup aml_dma "AML DMA"
 * @brief Management of low-level memory movements.
 *
 * AML DMA (inspired by Direct Memory Access engines) is an abstraction over the
 * ability to move data between places. A DMAs presents an interface that allows
 * clients to create an asynchronous request to move data and to wait for this
 * request to complete. Depending on the exact operation it is configured to do,
 * the DMA might transform the data during the operation.
 *
 * Implementations are mostly responsible for providing access to various types
 * of execution engine for data movement itself.
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
 * The request is in the format (dest layout, src layout)
 **/
#define AML_DMA_REQUEST_TYPE_LAYOUT 0

/**
 * The request is in the format (dest ptr, src ptr, size)
 */
#define AML_DMA_REQUEST_TYPE_PTR 1

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
	 * @param req[out]: the request handle to manage termination
	 *        of the movement.
	 * @param type: A valid AML_DMA_REQUEST_TYPE_* specifying the kind
	 *        of operation to perform.
	 * @param args: list of variadic arguments provided to aml_dma_copy()
	 * @return an AML error code.
	 **/
	int (*create_request)(struct aml_dma_data *dma,
			      struct aml_dma_request **req,
			      int type, va_list args);

	/**
	 * Destroy the request handle. If the data movement is still ongoing,
	 * then cancel it.
	 *
	 * @param dma: dma_implementation internal data.
	 * @param req: the request handle to manage termination of the movement.
	 * @return an AML error code.
	 **/
	int (*destroy_request)(struct aml_dma_data *dma,
			       struct aml_dma_request **req);

	/**
	 * Wait for termination of a data movement and destroy the request
	 * handle.
	 *
	 * @param dma: dma_implementation internal data.
	 * @param req: the request handle to manage termination of the movement.
	 * @return an AML error code.
	 **/
	int (*wait_request)(struct aml_dma_data *dma,
			    struct aml_dma_request **req);
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
 * Requests a synchronous data copy between two different buffers.
 * @param dma: an initialized DMA structure.
 * Variadic arguments: implementation-specific.
 * @return 0 if successful; an error code otherwise.
 **/
int aml_dma_copy(struct aml_dma *dma, int type, ...);

/**
 * Requests a data copy between two different buffers.This is an asynchronous
 * version of aml_dma_copy().
 * @param dma: an initialized DMA structure.
 * @param req: an address where the pointer to the newly assigned DMA request
 *        will be stored.
 * Variadic arguments: implementation-specific.
 * @return 0 if successful; an error code otherwise.
 **/
int aml_dma_async_copy(struct aml_dma *dma, struct aml_dma_request **req,
		       int type, ...);

/**
 * Waits for an asynchronous DMA request to complete.
 * @param dma: an initialized DMA structure.
 * @param req: a DMA request obtained using aml_dma_async_*() calls.
 * @return 0 if successful; an error code otherwise.
 **/
int aml_dma_wait(struct aml_dma *dma, struct aml_dma_request **req);

/**
 * Tears down an asynchronous DMA request before it completes.
 * @param dma: an initialized DMA structure.
 * @param req: a DMA request obtained using aml_dma_async_*() calls.
 * @return 0 if successful; an error code otherwise.
 **/
int aml_dma_cancel(struct aml_dma *dma, struct aml_dma_request **req);

/**
 * Generic helper to copy from one layout to another.
 * @param dst[out]: destination layout
 * @param src[in]: source layout
 */
int aml_copy_layout_generic(struct aml_layout *dst,
			    const struct aml_layout *src);


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
