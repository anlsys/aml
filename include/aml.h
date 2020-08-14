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
#include <excit.h>
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

#include "aml/utils/async.h"
#include "aml/utils/bitmap.h"
#include "aml/utils/error.h"
#include "aml/utils/features.h"
#include "aml/utils/inner-malloc.h"
#include "aml/utils/queue.h"
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
 * AML areas represent places where data can be stored.
 * In shared memory systems, locality is a major concern for performance.
 * Being able to query memory from specific places is of major interest
 * to achieve this goal. AML areas provide low-level mmap() / munmap() functions
 * to query memory from specific places materialized as areas. Available area
 * implementations dictate the way such places can be arranged and their
 * properties. It is important to note that the functions provided through the
 * Area API are low-level and are not optimized for performance as allocators
 * are.
 *
 * @image html area.png "Illustration of areas in a complex system." width=700cm
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
 * Opaque handle to pass additional options to area mmap hook.
 * This is implementation specific and cannot be used as a
 * generic interface but rather for customizing area behaviour
 * on per mmap basis.
 **/
struct aml_area_mmap_options;

/**
 * aml_area_ops is a structure containing implementations
 * of area operations.
 * Users may create or modify implementations by assembling
 * appropriate operations in such a structure.
 **/
struct aml_area_ops {
	/**
	 * Building block for coarse grain allocator of virtual memory.
	 *
	 * @param[in] data: Opaque handle to implementation specific data.
	 * @param[in] size: The minimum size of allocation.
	 *        Is greater than 0. Must not fail unless not enough
	 *        memory is available, or ptr argument does not point to a
	 *        suitable address.
	 *        In case of failure, aml_errno must be set to an appropriate
	 *        value.
	 * @param opts: Opaque handle to pass additional options to area
	 *        mmap hook. Can be NULL and must work with NULL opts.
	 * @return a pointer to allocated memory object.
	 **/
	void* (*mmap)(const struct aml_area_data  *data,
		      size_t                       size,
		      struct aml_area_mmap_options *opts);

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

	/**
	 * Print the implementation-specific information available
	 * @param stream the stream to print to
	 * @param prefix a prefix string to use on all lines
	 * @param data non-NULL handle to area internal data.
	 * @return 0 if successful, an error code otherwise.
	 **/
	int (*fprintf)(const struct aml_area_data *data,
		       FILE *stream, const char *prefix);
};

/**
 * An AML area is an implementation of memory operations for several types
 * of devices through a consistent abstraction.
 * This abstraction is meant to be implemented for several kinds of devices,
 * i.e., the same function calls allocate different kinds of devices depending
 * on the area implementation provided.
 **/
struct aml_area {
	/** Basic memory operations implementation **/
	struct aml_area_ops *ops;
	/** Implementation specific data. Set to NULL at creation. **/
	struct aml_area_data *data;
};

/**
 * Low-level function for obtaining memory from an area.
 * @param[in] area: A valid area implementing access to the target memory.
 * @param[in] size: The usable size of memory to obtain.
 * @param[in, out] opts: Opaque handle to pass additional options to the area.
 * @return a pointer to the memory range of the requested size allocated
 * within the area.
 * @return NULL on failure, with aml_errno set to the appropriate error
 * code.
 **/
void *aml_area_mmap(const struct aml_area        *area,
		    size_t                        size,
		    struct aml_area_mmap_options *opts);

/**
 * Releases memory region obtained with aml_area_mmap().
 * @param area: A valid area implementing access to the target memory.
 * @param ptr: A pointer to the memory obtained with aml_area_mmap()
 *        using the same "area" and "size" parameters.
 * @param size: The size of the memory region pointed to by "ptr".
 * @return 0 if successful, an error code otherwise.
 * @see aml_area_mmap()
 **/
int
aml_area_munmap(const struct aml_area *area,
		void                  *ptr,
		size_t                 size);

/**
 * Print on the file handle the metadata associated with this area.
 * @param stream the stream to print on
 * @param prefix prefix to use on all lines
 * @param area area to print
 * @return 0 if successful, an error code otherwise.
 */
int aml_area_fprintf(FILE *stream, const char *prefix,
		     const struct aml_area *area);

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
 * For a definition of row and columns of matrices see :
 * https://en.wikipedia.org/wiki/Matrix_(mathematics)
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
 * However, internally, the library will always store dimensions in such a way
 * that elements along the first dimension
 * are contiguous in memory. This order is defined with the value
 * AML_LAYOUT_ORDER_COLUMN_MAJOR (AML_LAYOUT_ORDER_FORTRAN). See:
 * https://en.wikipedia.org/wiki/Row-_and_column-major_order
 * Additionally, AML provides access to elements without the overhead of user
 * order choice through function suffixed with "native".
 * @see aml_layout_deref()
 * @see aml_layout_deref_native()
 * @see aml_layout_dims_native()
 * @see aml_layout_slice_native()
 *
 * The layout abstraction also provides a function to reshape data
 * with a different set of dimensions. A reshaped layout will access
 * the same data but with different coordinates as depicted in the
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
	 * Function to retrieve a pointer to the start of the actual memory
	 * buffer under this layout.
	 * @param data[in] the non-NULL handle to layout internal data.
	 * @return a pointer to the buffer on success
	 * @return NULL on failure, with aml_errno set to the error reason.
	 **/
	void *(*rawptr)(const struct aml_layout_data *data);

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
		     const size_t *offsets,
		     const size_t *dims,
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
			    const size_t *offsets,
			    const size_t *dims,
			    const size_t *strides);
	/**
	 * Print the implementation-specific information available on a layout,
	 * content excluded.
	 * @param stream the stream to print to
	 * @param prefix a prefix string to use on all lines
	 * @param data non-NULL handle to layout internal data.
	 * @return 0 if successful, an error code otherwise.
	 **/
	int (*fprintf)(const struct aml_layout_data *data,
		       FILE *stream, const char *prefix);

	/**
	 * Duplicate a layout (does not copy data, but deep copy
	 * metadata).
	 * If the layout relies on sublayouts (e.g. pad, reshape), those will be
	 * copied too.
	 * @param[in] layout a non-NULL handle to a layout to copy.
	 * @param[out] out a pointer to where to store the new layout.
	 * @return -AML_ENOTSUP if operation is not available.
	 * @return -AML_ENOMEM if layout allocation failed.
	 * @return -AML_EINVAL if src or dest are NULL.
	 * @return AML_SUCCESS if copy succeeded.
	 **/
	int (*duplicate)(const struct aml_layout *layout,
	                 struct aml_layout **out);

	/**
	 * Destroys the layout and frees all associated memory.
	 **/
	void (*destroy)(struct aml_layout *);
};

/**
 * Tag specifying user storage of dimensions inside a layout.
 * Layout order is the first bit in an integer bitmask.
 * @see AML_LAYOUT_ORDER()
 * This tag will store dimensions in the order provided by the user,
 * i.e., elements of the last dimension will be contiguous in memory.
 **/
#define AML_LAYOUT_ORDER_FORTRAN (0<<0)

/**
 * Tag specifying user storage of dimensions inside a layout.
 * Layout order is the first bit in an integer bitmask.
 * @see AML_LAYOUT_ORDER()
 * This tag will store dimensions in the reverse order to the one provided
 * by the user, i.e., elements of the first dimension will be contiguous
 * in memory. This storage is the actual storage used by the library
 * inside the structure.
 **/
#define AML_LAYOUT_ORDER_C (1<<0)

/**
 * This is equivalent to AML_LAYOUT_ORDER_FORTRAN.
 * @see AML_LAYOUT_ORDER_FORTRAN
 **/
#define AML_LAYOUT_ORDER_COLUMN_MAJOR (0<<0)

/**
 * This is equivalent to AML_LAYOUT_ORDER_C.
 * @see AML_LAYOUT_ORDER_C
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
 * @param[in] layout: An initialized layout.
 * @param[in] coords: The coordinates on which to access data.
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
 * Return a pointer to the first byte of the buffer this layout maps to.
 * @param layout an initialized layout
 * @return a raw pointer to the start of the layout, NULL on error.
 */
void *aml_layout_rawptr(const struct aml_layout *layout);

/**
 * Get the order in which dimensions of the layout are supposed to be
 * accessed by the user.
 * @param[in] layout: An initialized layout.
 * @return The order (>0) on success, an AML error (<0) on failure.
 * @return On success, a bitmask with order bit set (or not set).
 * Output value can be further checked against order AML_LAYOUT_ORDER
 * flags by using the macro AML_LAYOUT_ORDER() on the output value.
 * @see AML_LAYOUT_ORDER()
 **/
int aml_layout_order(const struct aml_layout *layout);

/**
 * Return the layout dimensions in the user order.
 * @param[in] layout: An initialized layout.
 * @param[out] dims: A non-NULL array of dimensions to fill. It is
 * supposed to be large enough to contain aml_layout_ndims() elements.
 * @return AML_SUCCESS on success, else an AML error code.
 **/
int aml_layout_dims(const struct aml_layout *layout, size_t *dims);

/**
 * Return the number of dimensions in a layout.
 * @param[in] layout: An initialized layout.
 * @return The number of dimensions in the layout.
 **/
size_t aml_layout_ndims(const struct aml_layout *layout);

/**
 * @brief Return the size of layout elements.
 * @param[in] layout: An initialized layout.
 * @return The size of elements stored with this layout.
 **/
size_t aml_layout_element_size(const struct aml_layout *layout);

/**
 * @brief Reshape the layout with different dimensions.
 * This function checks that the number of elements of
 * the reshaped layout matches the number of elements
 * in the original layout. Additional constraint may apply
 * depending on the layout implementation.
 * @param[in] layout: An initialized layout.
 * @param[out] reshaped_layout: A newly allocated layout
 * with the queried shape on succes.
 * @param[in] ndims: The number of dimensions of the new layout.
 * @param[in] dims: The number of elements along each dimension of
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
 * This function checks that the number of elements along
 * each dimension of the slice actually fits in the original
 * layout.
 * @param[in] layout: An initialized layout.
 * @param[out] reshaped_layout: a pointer where to store the address of a
 * newly allocated layout with the queried subset of the
 * original layout on succes.
 * @param[in] dims: The number of elements of the slice along each
 * dimension.
 * @param[in] offsets: The index of the first element of the slice
 * in each dimension. If NULL, offset is set to 0.
 * @param[in] strides: The displacement (in number of elements) between
 * elements of the slice. If NULL, stride is set to 1.
 * @return AML_SUCCESS on success, else an AML error code (<0).
 **/
int aml_layout_slice(const struct aml_layout *layout,
		     struct aml_layout **reshaped_layout,
		     const size_t *offsets,
		     const size_t *dims,
		     const size_t *strides);

/**
 * Print on the file handle the metadata associated with this layout.
 * @param stream the stream to print on
 * @param prefix prefix to use on all lines
 * @param layout layout to print
 * @return 0 if successful, an error code otherwise.
 */
int aml_layout_fprintf(FILE *stream, const char *prefix,
		       const struct aml_layout *layout);

/**
 * Creates a duplicate of the layout (independent deep copy of all its metadata,
 * no user data is actually copied).
 * @param[in] src the layout to duplicate
 * @param[out] out a pointer to where to store the new layout
 * @return -AML_ENOMEM if layout allocation failed.
 * @return -AML_EINVAL if src or dest are NULL.
 * @return AML_SUCCESS if copy succeeded.
 **/
int aml_layout_duplicate(const struct aml_layout *src, struct aml_layout **out);

/**
 * Destroy (free) a layout, irrespective of its type.
 * @param[in,out] layout the layout to destroy. NULL on return.
 **/
void aml_layout_destroy(struct aml_layout **layout);

////////////////////////////////////////////////////////////////////////////////

/**
 * @}
 * @defgroup aml_tiling "AML Tiling"
 * @brief Tiling Data Structure High-Level API
 *
 * Tiling is a representation of the decomposition of data structures. It
 * identifies ways a layout can be split into layouts of smaller size. As such,
 * the main function of a tiling is to provide an index into subcomponents of a
 * layout. Implementations focus on the ability to provide sublayouts of
 * different sizes at the corners, and linearization of the index range.
 * @{
 **/

////////////////////////////////////////////////////////////////////////////////

/**
 * Tag specifying user storage of dimensions inside a layout.
 * Layout order is the first bit in an integer bitmask.
 * @see AML_TILING_ORDER()
 * This tag will store dimensions in the order provided by the user,
 * i.e elements of the last dimension will be contiguous in memory.
 **/
#define AML_TILING_ORDER_FORTRAN (0<<0)

/**
 * Tag specifying user storage of dimensions inside a layout.
 * Layout order is the first bit in an integer bitmask.
 * @see AML_TILING_ORDER()
 * This tag will store dimensions in the reversed order provided
 * by the user, i.e elements of the first dimension will be contiguous
 * in memory. This storage is the actual storage used by the library
 * inside the structure.
 **/
#define AML_TILING_ORDER_C (1<<0)

/**
 * This is equivalent to AML_TILING_ORDER_FORTRAN.
 * @see AML_TILING_ORDER_FORTRAN
 **/
#define AML_TILING_ORDER_COLUMN_MAJOR (0<<0)

/**
 * This is equivalent to AML_TILING_ORDER_C.
 * @see AML_TILING_ORDER_C
 **/
#define AML_TILING_ORDER_ROW_MAJOR (1<<0)

/**
 * Get the order bit of an integer bitmask.
 * The value can be further checked for equality
 * with AML_TILING_ORDER_* values.
 * @param x: An integer with the first bit set
 * to the order value.
 * @return An integer containing only the bit order.
 **/
#define AML_TILING_ORDER(x) ((x) & (1<<0))

/**
 * aml_tiling_data is an opaque handle defined by each aml_tiling
 * implementation. This not supposed to be used by end users.
 **/
struct aml_tiling_data;

/**
 * aml_tiling_ops is a structure containing a set of operation
 * over a tiling. These operations focus on:
 * - retrieving a tile
 * - getting information about the size of tiles and the tiling itself.
 **/
struct aml_tiling_ops {
	/** retrieve a tile as a layout **/
	struct aml_layout* (*index)(const struct aml_tiling_data *t,
				    const size_t *coords);
	/** retrieve a tile as a layout with coordinates in native order  **/
	struct aml_layout* (*index_native)(const struct aml_tiling_data *t,
					   const size_t *coords);
	void *(*rawptr)(const struct aml_tiling_data *t,
			const size_t *coords);
	int (*tileid)(const struct aml_tiling_data *t, const size_t *coords);
	int (*order)(const struct aml_tiling_data *t);
	int (*tile_dims)(const struct aml_tiling_data *t, size_t *dims);
	int (*dims)(const struct aml_tiling_data *t, size_t *dims);
	int (*dims_native)(const struct aml_tiling_data *t, size_t *dims);
	size_t (*ndims)(const struct aml_tiling_data *t);
	size_t (*ntiles)(const struct aml_tiling_data *t);

	/**
	 * Print the implementation-specific information available on a tiling,
	 * content excluded.
	 * @param stream the stream to print to
	 * @param prefix a prefix string to use on all lines
	 * @param data non-NULL handle to tiling internal data.
	 * @return 0 if successful, an error code otherwise.
	 **/
	int (*fprintf)(const struct aml_tiling_data *data,
		       FILE *stream, const char *prefix);
};

/**
 **/
struct aml_tiling {
	/** @see aml_tiling_ops **/
	struct aml_tiling_ops *ops;
	/** @see aml_tiling_data **/
	struct aml_tiling_data *data;
};

/**
 * Get the order in which dimensions of the tiling are supposed to be
 * accessed by the user.
 * @param[in] tiling: An initialized tiling.
 * @return The order (>0) on success, an AML error (<0) on failure.
 * @return On success, a bitmask with order bit set (or not set).
 * Output value can be further checked against order AML_TILING_ORDER
 * flags by using the macro AML_TILING_ORDER() on the output value.
 * @see AML_TILING_ORDER()
 **/
int aml_tiling_order(const struct aml_tiling *tiling);

/**
 * Return the tiling dimensions in the user order.
 * @param[in] tiling: An initialized tiling.
 * @param[out] dims: A non-NULL array of dimensions to fill. It is
 * supposed to be large enough to contain aml_tiling_ndims() elements.
 * @return AML_SUCCESS on success, else an AML error code.
 **/
int aml_tiling_dims(const struct aml_tiling *tiling, size_t *dims);

/**
 * Return the dimensions of a tile in the tiling, in the user order.
 * @param[in] tiling: An initialized tiling.
 * @param[out] dims: A non-NULL array of dimensions to fill. It is
 * supposed to be large enough to contain aml_tiling_ndims() elements.
 * @return AML_SUCCESS on success, else an AML error code.
 **/
int aml_tiling_tile_dims(const struct aml_tiling *tiling, size_t *dims);

/**
 * Provide the number of dimensions in a tiling.
 * @param tiling: an initialized tiling structure.
 * @return the number of dimensions in the tiling.
 **/
size_t aml_tiling_ndims(const struct aml_tiling *tiling);

/**
 * Provide the number of tiles in a tiling.
 * @param tiling: an initialized tiling structure.
 * @return the number of tiles in the tiling.
 **/
size_t aml_tiling_ntiles(const struct aml_tiling *tiling);

/**
 * Return the tile at specified coordinates in the tiling
 * @param tiling: an initialized tiling
 * @param coords: the coordinates for the tile
 * @return the tile as a layout on success, NULL on error.
 **/
struct aml_layout *aml_tiling_index(const struct aml_tiling *tiling,
				    const size_t *coords);

/**
 * Return a pointer to the first valid coordinate in the underlying tile.
 * @param tiling an initialized tiling
 * @param coords the coordinates for the tile
 * @return a raw pointer to the start of the buffer for a tile, NULL on error.
 */
void *aml_tiling_rawptr(const struct aml_tiling *tiling, const size_t *coords);

/**
 * Return a unique identifier for a tile based on coordinates in the tiling
 * @param tiling: an initialized tiling
 * @param coords: the coordinates for the tile
 * @return a uuid for the tile.
 */
int aml_tiling_tileid(const struct aml_tiling *tiling, const size_t *coords);

/**
 * Return the tile with this identifier
 * @param tiling: an initialized tiling
 * @param uuid: a unique identifier for this tile
 * @return the tiling as a layout on success, NULL on error.
 */
struct aml_layout *aml_tiling_index_byid(const struct aml_tiling *tiling,
					 const int uuid);

/**
 * Print on the file handle the metadata associated with this tiling.
 * @param stream the stream to print on
 * @param prefix prefix to use on all lines
 * @param tiling tiling to print
 * @return 0 if successful, an error code otherwise.
 */
int aml_tiling_fprintf(FILE *stream, const char *prefix,
		       const struct aml_tiling *tiling);

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
 * Type of the function used to perform the DMA between two layouts.
 * @param dst: destination layout
 * @param src: source layout
 * @param arg: extra argument needed by the operator
 **/
typedef int (*aml_dma_operator)(struct aml_layout *dst,
				const struct aml_layout *src, void *arg);

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
	 * @param dest: layout describing the destination.
	 * @param src: layout describing the source.
	 * @return an AML error code.
	 **/
	int (*create_request)(struct aml_dma_data *dma,
			      struct aml_dma_request **req,
			      struct aml_layout *dest,
			      struct aml_layout *src,
			      aml_dma_operator op, void *op_arg);

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

	/**
	 * Print the implementation-specific information available on a dma.
	 * @param stream the stream to print to
	 * @param prefix a prefix string to use on all lines
	 * @param data non-NULL handle to dma internal data.
	 * @return 0 if successful, an error code otherwise.
	 **/
	int (*fprintf)(const struct aml_dma_data *data,
		       FILE *stream, const char *prefix);
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
 *
 * Layouts are copied internally if necessary, avoiding the need for users to
 * keep the layouts alive during the request.
 *
 * @param dma: an initialized DMA structure.
 * @param dest: layout describing the destination.
 * @param src: layout describing the source.
 * @param op: optional custom operator for this dma
 * @param op_arg: optional argument to the operator
 * @return 0 if successful; an error code otherwise.
 **/
int aml_dma_copy_custom(struct aml_dma *dma, struct aml_layout *dest,
		 struct aml_layout *src, aml_dma_operator op, void *op_arg);

/**
 * Requests a data copy between two different buffers.This is an asynchronous
 * version of aml_dma_copy().
 *
 * Layouts are copied internally if necessary, avoiding the need for users to
 * keep the layouts alive during the request.
 *
 * @param dma: an initialized DMA structure.
 * @param req: an address where the pointer to the newly assigned DMA request
 *        will be stored.
 * @param dest: layout describing the destination.
 * @param src: layout describing the source.
 * @param op: optional custom operator for this dma
 * @param op_arg: optional argument to the operator
 * @return 0 if successful; an error code otherwise.
 **/
int aml_dma_async_copy_custom(struct aml_dma *dma, struct aml_dma_request **req,
		       struct aml_layout *dest,
		       struct aml_layout *src,
		       aml_dma_operator op, void *op_arg);

#define aml_dma_copy(dma, d, s) aml_dma_copy_custom(dma, d, s, NULL, NULL)
#define aml_dma_async_copy(dma, r, d, s) \
	aml_dma_async_copy_custom(dma, r, d, s, NULL, NULL)

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
 * Print on the file handle the metadata associated with this dma.
 * @param stream the stream to print on
 * @param prefix prefix to use on all lines
 * @param dma dma to print
 * @return 0 if successful, an error code otherwise.
 */
int aml_dma_fprintf(FILE *stream, const char *prefix,
		    const struct aml_dma *dma);

/**
 * Generic helper to copy from one layout to another.
 * @param[out] dst: destination layout
 * @param[in] src: source layout
 * @param[in] arg: unused (should be NULL)
 */
int aml_copy_layout_generic(struct aml_layout *dst,
			    const struct aml_layout *src, void *arg);

int aml_copy_layout_transform_generic(struct aml_layout *dst,
				      const struct aml_layout *src,
				      const size_t *target_dims);

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
			      struct aml_layout **dest, int *destid,
			      struct aml_layout *src, int srcid);
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
 * @param dest: destination layout (on the scratch)
 * @param scratchid: an argument of type int*; gets filled with the scratch tile
 *        identifier where the data will be pulled into.
 * @param src: the source layout.
 * @param srcid: an argument of type int; the source tile identifier.
 * @see aml_scratch_baseptr()
 * @return 0 if successful; an error code otherwise.
 **/
int aml_scratch_pull(struct aml_scratch *scratch,
		     struct aml_layout **dest, int *scratchid,
		     struct aml_layout *src, int srcid);

/**
 * Requests a pull from regular memory to the scratchpad. This is an
 * asynchronous version of aml_scratch_pull().
 * @param scratch: an initialized scratchpad structure.
 * @param req: an address where the pointer to the newly assigned scratch
 *        request will be stored.
 * @param scratch_layout: the layout on the scratch
 * @param scratchid: an argument of type int*; gets filled with the scratch tile
 *        identifier where the data will be pulled into.
 * @param src_layout: the source layout to pull.
 * @param srcid: an argument of type int; the source tile identifier.
 * @return 0 if successful; an error code otherwise.
 * @see aml_scratch_pull()
 **/
int aml_scratch_async_pull(struct aml_scratch *scratch,
			   struct aml_scratch_request **req,
			   struct aml_layout **scratch_layout, int *scratchid,
			   struct aml_layout *src_layout, int srcid);
/**
 * Requests a synchronous push from the scratchpad to regular memory.
 * @param scratch: an initialized scratchpad structure.
 * @param dest_layout: the destination layout
 * @param destid: an argument of type int*; gets filled with the destination
 * tile identifier where the data will be pushed into (and where it was pulled
 * from in the first place).
 * @param scratch_layout: the source layout on the scratch
 * @param scratchid: an argument of type int; the scratchpad tile identifier.
 * @return 0 if successful; an error code otherwise.
 * @see aml_scratch_baseptr()
 **/
int aml_scratch_push(struct aml_scratch *scratch,
		     struct aml_layout **dest_layout, int *destid,
		     struct aml_layout *scratch_layout, int scratchid);

/**
 * Requests a push from the scratchpad to regular memory.  This is an
 * asynchronous version of aml_scratch_push().
 * @param scratch: an initialized scratchpad structure.
 * @param req: an address where the pointer to the newly assigned scratch
 *        request will be stored.
 * @param dest_layout: the destination layout
 * @param destid: an argument of type int*; gets filled with the destination
 * tile identifier where the data will be pushed into (and where it was pulled
 * from in the first place).
 * @param scratch_layout: the source layout on the scratch
 * @param scratchid: an argument of type int; the scratchpad tile identifier.
 * @return 0 if successful; an error code otherwise.
 * @see aml_scratch_push()
 **/
int aml_scratch_async_push(struct aml_scratch *scratch,
			   struct aml_scratch_request **req,
			   struct aml_layout **dest_layout, int *destid,
			   struct aml_layout *scratch_layout, int scratchid);
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
