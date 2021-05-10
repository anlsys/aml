/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
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

#ifdef __cplusplus
extern "C" {
#endif

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
 * Initialize the library.
 * @param[inout] argc: pointer to the main()'s argc argument; contents can get
 *        modified.
 * @param[inout] argv: pointer to the main()'s argv argument; contents can get
 *        modified.
 * @return 0 if successful; an error code otherwise.
 **/
int aml_init(int *argc, char **argv[]);

/**
 * Terminate the library.
 * @return 0 if successful; an error code otherwise.
 **/
int aml_finalize(void);

////////////////////////////////////////////////////////////////////////////////

/**
 * @}
 * @defgroup aml_area "AML Area"
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
	 * @param[in] data: opaque handle to implementation specific data.
	 * @param[in] size: the minimum size of allocation.
	 *        Is greater than 0. Must not fail unless not enough
	 *        memory is available, or ptr argument does not point to a
	 *        suitable address.
	 *        In case of failure, aml_errno must be set to an appropriate
	 *        value.
	 * @param[in] opts: opaque handle to pass additional options to area
	 *        mmap hook. Can be NULL and must work with NULL opts.
	 * @return a pointer to allocated memory object.
	 **/
	void *(*mmap)(const struct aml_area_data *data,
	              size_t size,
	              struct aml_area_mmap_options *opts);

	/**
	 * Building block for unmapping of virtual memory mapped with mmap()
	 * of the same area.
	 *
	 * @param[in] data: an opaque handle to implementation specific data.
	 * @param[in] ptr: pointer to data mapped in physical memory. Cannot be
	 *        NULL.
	 * @param[in] size: the size of data. Cannot be 0.
	 * @return: AML_AREA_* error code.
	 * @see mmap()
	 **/
	int (*munmap)(const struct aml_area_data *data, void *ptr, size_t size);

	/**
	 * Print the implementation-specific information available
	 * @param[in] stream: the stream to print to
	 * @param[in] prefix: a prefix string to use on all lines
	 * @param[in] data: non-NULL handle to area internal data.
	 * @return 0 if successful, an error code otherwise.
	 **/
	int (*fprintf)(const struct aml_area_data *data,
	               FILE *stream,
	               const char *prefix);
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
 * @param[in] area: a valid area implementing access to the target memory.
 * @param[in] size: the usable size of memory to obtain.
 * @param[in, out] opts: opaque handle to pass additional options to the area.
 * @return a pointer to the memory range of the requested size allocated
 * within the area ; NULL on failure, with aml_errno set to the appropriate
 * error code.
 **/
void *aml_area_mmap(const struct aml_area *area,
                    size_t size,
                    struct aml_area_mmap_options *opts);

/**
 * Release memory region obtained with aml_area_mmap().
 * @param[in] area: a valid area implementing access to the target memory.
 * @param[in, out] ptr: a pointer to the memory obtained with aml_area_mmap()
 *        using the same "area" and "size" parameters.
 * @param[in] size: the size of the memory region pointed to by "ptr".
 * @return 0 if successful, an error code otherwise.
 * @see aml_area_mmap()
 **/
int aml_area_munmap(const struct aml_area *area, void *ptr, size_t size);

/**
 * Print on the file handle the metadata associated with this area.
 * @param[in] stream: the stream to print on
 * @param[in] prefix: prefix to use on all lines
 * @param[in] area: area to print
 * @return 0 if successful, an error code otherwise.
 */
int aml_area_fprintf(FILE *stream,
                     const char *prefix,
                     const struct aml_area *area);

////////////////////////////////////////////////////////////////////////////////

/**
 * @}
 * @defgroup aml_layout "AML Layout"
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
	 * @param data[in]: the non-NULL handle to layout internal data.
	 * @param coords[in]: the non-NULL coordinates on which to access data.
	 * Coordinates are checked to be valid in aml_layout_deref().
	 * @return a pointer to the dereferenced element on success.
	 * @return NULL on failure with aml_errno set to the error reason.
	 **/
	void *(*deref)(const struct aml_layout_data *data,
	               const size_t *coords);

	/**
	 * Function for derefencing elements of a layout inside the library.
	 * Layout assumes data is always stored in AML_LAYOUT_ORDER_FORTRAN
	 * order. Coordinates provided by the library will match the same
	 * order, i.e last dimension first.
	 * @param data[in]: the non-NULL handle to layout internal data.
	 * @param coords[in]: the non-NULL coordinates on which to access data.
	 * The first coordinate should be the last dimensions and so on to the
	 * last, coordinate, last dimension.
	 * @return a pointer to the dereferenced element on success.
	 * @return NULL on failure with aml_errno set to the error reason.
	 **/
	void *(*deref_native)(const struct aml_layout_data *data,
	                      const size_t *coords);

	/**
	 * Function to retrieve a pointer to the start of the actual memory
	 * buffer under this layout.
	 * @param[in] data: the non-NULL handle to layout internal data.
	 * @return a pointer to the buffer on success
	 * @return NULL on failure, with aml_errno set to the error reason.
	 **/
	void *(*rawptr)(const struct aml_layout_data *data);

	/**
	 * Get the order in which dimensions of the layout are
	 * supposed to be accessed by the user.
	 * @param[in] data: the non-NULL handle to layout internal data.
	 * @return order value. It is a bitmask with order bit set (or not set).
	 * Output value can be further checked against order AML_LAYOUT_ORDER
	 * flags by using the macro AML_LAYOUT_ORDER() on output value.
	 * @see AML_LAYOUT_ORDER()
	 **/
	int (*order)(const struct aml_layout_data *data);

	/**
	 * Return the layout dimensions in the user order.
	 * @param[in] data: the non-NULL handle to layout internal data.
	 * @param[out] dims: the non-NULL array of dimensions to fill. It is
	 * supposed to be large enough to contain ndims() elements.
	 * @return AML_SUCCESS on success, else an AML error code.
	 **/
	int (*dims)(const struct aml_layout_data *data, size_t *dims);

	/**
	 * Return the layout dimensions in the order they are actually stored
	 * in the library.
	 * @param[in] data: the non-NULL handle to layout internal data.
	 * @param[out] dims: the non-NULL array of dimensions to fill. It is
	 * supposed to be large enough to contain ndims() elements.
	 * @return AML_SUCCESS on success, else an AML error code.
	 **/
	int (*dims_native)(const struct aml_layout_data *data, size_t *dims);

	/**
	 * Return the number of dimensions in a layout.
	 * @param[in] data: the non-NULL handle to layout internal data.
	 * @return the number of dimensions in the layout.
	 **/
	size_t (*ndims)(const struct aml_layout_data *data);

	/**
	 * Return the size of layout elements.
	 * @param[in] data: the non-NULL handle to layout internal data.
	 * @return the size of elements stored with this layout.
	 **/
	size_t (*element_size)(const struct aml_layout_data *data);

	/**
	 * Reshape the layout with different dimensions.
	 * Layout dimensions are checked in aml_layout_reshape() to store
	 * the exact same number of elements.
	 * @param[in] data: the non-NULL handle to layout internal data.
	 * @param[out] output: a non NULL pointer to a layout where to allocate
	 * a new layout resulting from the reshape operation.
	 * @param[in] ndims: the number of dimensions of the new layout.
	 * @param[in] dims: the number of elements along each dimension of
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
	 * @param[in] data: the non-NULL handle to layout internal data.
	 * @param[out] output: a non NULL pointer to a layout where to allocate
	 * a new layout resulting from the slice operation.
	 * @param[in] dims: the number of elements of the slice along each
	 * dimension .
	 * @param[in] offsets: the index of the first element of the slice
	 * in each dimension.
	 * @param[in] strides: the displacement (in number of elements) between
	 * elements of the slice.
	 * @return a newly allocated layout with the queried subset of the
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
	 * @param[in] data: the non-NULL handle to layout internal data.
	 * @param[out] output: a non NULL pointer to a layout where to allocate
	 * a new layout resulting from the slice operation.
	 * @param[in] dims: the number of elements of the slice along each
	 * dimension .
	 * @param[in] offsets: the index of the first element of the slice
	 * in each dimension.
	 * @param[in] strides: the displacement (in number of elements) between
	 * elements of the slice.
	 * @return a newly allocated layout with the queried subset of the
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
	 * @param[in] stream: the stream to print to
	 * @param[in] prefix: a prefix string to use on all lines
	 * @param[in] data: non-NULL handle to layout internal data.
	 * @return 0 if successful, an error code otherwise.
	 **/
	int (*fprintf)(const struct aml_layout_data *data,
	               FILE *stream,
	               const char *prefix);

	/**
	 * Duplicate a layout (does not copy data, but deep copy
	 * metadata).
	 * If the layout relies on sublayouts (e.g. pad, reshape), those will be
	 * copied too.
	 * @param[in] layout: a non-NULL handle to a layout to copy.
	 * @param[out] out: a pointer to where to store the new layout.
	 * @param[in] ptr: if not NULL use this pointer as the new layout raw
	 *pointer.
	 * @return -AML_ENOTSUP if operation is not available.
	 * @return -AML_ENOMEM if layout allocation failed.
	 * @return -AML_EINVAL if src or dest are NULL.
	 * @return AML_SUCCESS if copy succeeded.
	 **/
	int (*duplicate)(const struct aml_layout *layout,
	                 struct aml_layout **out,
	                 void *ptr);

	/**
	 * Destroy the layout and frees all associated memory.
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
#define AML_LAYOUT_ORDER_FORTRAN (0 << 0)

/**
 * Tag specifying user storage of dimensions inside a layout.
 * Layout order is the first bit in an integer bitmask.
 * @see AML_LAYOUT_ORDER()
 * This tag will store dimensions in the reverse order to the one provided
 * by the user, i.e., elements of the first dimension will be contiguous
 * in memory. This storage is the actual storage used by the library
 * inside the structure.
 **/
#define AML_LAYOUT_ORDER_C (1 << 0)

/**
 * This is equivalent to AML_LAYOUT_ORDER_FORTRAN.
 * @see AML_LAYOUT_ORDER_FORTRAN
 **/
#define AML_LAYOUT_ORDER_COLUMN_MAJOR (0 << 0)

/**
 * This is equivalent to AML_LAYOUT_ORDER_C.
 * @see AML_LAYOUT_ORDER_C
 **/
#define AML_LAYOUT_ORDER_ROW_MAJOR (1 << 0)

/**
 * Get the order bit of an integer bitmask.
 * The value can be further checked for equality
 * with AML_LAYOUT_ORDER_* values.
 * @param[in] x: an integer with the first bit set
 * to the order value.
 * @return an integer containing only the bit order.
 **/
#define AML_LAYOUT_ORDER(x) ((x) & (1 << 0))

/**
 * Dereference an element of a layout by its coordinates.
 * @param[in] layout: an initialized layout.
 * @param[in] coords: the coordinates on which to access data.
 * @return a pointer to the dereferenced element on success ; NULL on failure
 * with aml_errno set to the error reason:
 * * AML_EINVAL if coordinate are out of bound
 * * See specific implementation of layout for further information
 * on possible error codes.
 **/
void *aml_layout_deref(const struct aml_layout *layout, const size_t *coords);

/**
 * Equivalent to aml_layout_deref() but with bound checking
 * on coordinates.
 * @param[in] layout: an initialized layout.
 * @param[in] coords: the coordinates on which to access data.
 * @return a pointer to dereferenced element on success ; NULL on failure
 * with aml_errno set to the error reason.
 * @see aml_layout_deref()
 **/
void *aml_layout_deref_safe(const struct aml_layout *layout,
                            const size_t *coords);

/**
 * Return a pointer to the first byte of the buffer this layout maps to.
 * @param[in] layout: an initialized layout
 * @return a raw pointer to the start of the layout, NULL on error.
 */
void *aml_layout_rawptr(const struct aml_layout *layout);

/**
 * Get the order in which dimensions of the layout are supposed to be
 * accessed by the user.
 * @param[in] layout: an initialized layout.
 * @return a bitmask with order bit on success, an AML error (<0) on failure.
 *	Output value can be further checked against order AML_LAYOUT_ORDER
 *	flags by using the macro AML_LAYOUT_ORDER() on the output value.
 * @see AML_LAYOUT_ORDER()
 **/
int aml_layout_order(const struct aml_layout *layout);

/**
 * Return the layout dimensions in the user order.
 * @param[in] layout: an initialized layout.
 * @param[out] dims: a non-NULL array of dimensions to fill. It is
 * supposed to be large enough to contain aml_layout_ndims() elements.
 * @return 0 on success, else an AML error code.
 **/
int aml_layout_dims(const struct aml_layout *layout, size_t *dims);

/**
 * Provide the number of dimensions in the layout.
 * @param[in] layout: an initialized layout.
 * @return the number of dimensions in the layout.
 **/
size_t aml_layout_ndims(const struct aml_layout *layout);

/**
 * Return the size of layout elements.
 * @param[in] layout: an initialized layout.
 * @return the size of elements stored with this layout.
 **/
size_t aml_layout_element_size(const struct aml_layout *layout);

/**
 * Reshape the layout with different dimensions.
 * This function checks that the number of elements of
 * the reshaped layout matches the number of elements
 * in the original layout. Additional constraint may apply
 * depending on the layout implementation.
 * @param[in] layout: an initialized layout.
 * @param[out] reshaped_layout: a newly allocated layout
 * with the queried shape on succes.
 * @param[in] ndims: the number of dimensions of the new layout.
 * @param[in] dims: the number of elements along each dimension of
 * the new layout.
 * @return 0 on success, an AML error code otherwise.
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
 * @param[in] layout: an initialized layout.
 * @param[out] reshaped_layout: a pointer where to store the address of a
 * newly allocated layout with the queried subset of the
 * original layout on succes.
 * @param[in] dims: the number of elements of the slice along each
 * dimension.
 * @param[in] offsets: the index of the first element of the slice
 * in each dimension. If NULL, offset is set to 0.
 * @param[in] strides: the displacement (in number of elements) between
 * elements of the slice. If NULL, stride is set to 1.
 * @return 0 on success, else an AML error code (<0).
 **/
int aml_layout_slice(const struct aml_layout *layout,
                     struct aml_layout **reshaped_layout,
                     const size_t *offsets,
                     const size_t *dims,
                     const size_t *strides);

/**
 * Print on the file handle the metadata associated with this layout.
 * @param[in] stream: the stream to print on
 * @param[in] prefix: prefix to use on all lines
 * @param[in] layout: layout to print
 * @return 0 if successful, an AML error code otherwise.
 */
int aml_layout_fprintf(FILE *stream,
                       const char *prefix,
                       const struct aml_layout *layout);

/**
 * Create a duplicate of the layout (independent deep copy of all its metadata,
 * no user data is actually copied).
 * @param[in] src: the layout to duplicate
 * @param[out] out: a pointer to where to store the new layout
 * @param[in] ptr: if not NULL use this pointer as the new layout raw pointer.
 * @return -AML_ENOMEM if layout allocation failed.
 * @return -AML_EINVAL if src or dest are NULL.
 * @return AML_SUCCESS if copy succeeded.
 **/
int aml_layout_duplicate(const struct aml_layout *src,
                         struct aml_layout **out,
                         void *ptr);

/**
 * Destroy (free) a layout, irrespective of its type.
 * @param[in,out] layout: the layout to destroy. NULL on return.
 **/
void aml_layout_destroy(struct aml_layout **layout);

////////////////////////////////////////////////////////////////////////////////

/**
 * @}
 * @defgroup aml_tiling "AML Tiling"
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
#define AML_TILING_ORDER_FORTRAN (0 << 0)

/**
 * Tag specifying user storage of dimensions inside a layout.
 * Layout order is the first bit in an integer bitmask.
 * @see AML_TILING_ORDER()
 * This tag will store dimensions in the reversed order provided
 * by the user, i.e elements of the first dimension will be contiguous
 * in memory. This storage is the actual storage used by the library
 * inside the structure.
 **/
#define AML_TILING_ORDER_C (1 << 0)

/**
 * This is equivalent to AML_TILING_ORDER_FORTRAN.
 * @see AML_TILING_ORDER_FORTRAN
 **/
#define AML_TILING_ORDER_COLUMN_MAJOR (0 << 0)

/**
 * This is equivalent to AML_TILING_ORDER_C.
 * @see AML_TILING_ORDER_C
 **/
#define AML_TILING_ORDER_ROW_MAJOR (1 << 0)

/**
 * Get the order bit of an integer bitmask.
 * The value can be further checked for equality
 * with AML_TILING_ORDER_* values.
 * @param[out] x: an integer with the first bit set
 * to the order value.
 * @return an integer containing only the bit order.
 **/
#define AML_TILING_ORDER(x) ((x) & (1 << 0))

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
	struct aml_layout *(*index)(const struct aml_tiling_data *t,
	                            const size_t *coords);
	/** retrieve a tile as a layout with coordinates in native order  **/
	struct aml_layout *(*index_native)(const struct aml_tiling_data *t,
	                                   const size_t *coords);
	void *(*rawptr)(const struct aml_tiling_data *t, const size_t *coords);
	int (*order)(const struct aml_tiling_data *t);
	int (*dims)(const struct aml_tiling_data *t, size_t *dims);
	int (*dims_native)(const struct aml_tiling_data *t, size_t *dims);
	size_t (*ndims)(const struct aml_tiling_data *t);
	size_t (*ntiles)(const struct aml_tiling_data *t);

	/**
	 * Print the implementation-specific information available on a tiling,
	 * content excluded.
	 * @param[in] stream: the stream to print to
	 * @param[in] prefix: a prefix string to use on all lines
	 * @param[in] data: non-NULL handle to tiling internal data.
	 * @return 0 if successful, an error code otherwise.
	 **/
	int (*fprintf)(const struct aml_tiling_data *data,
	               FILE *stream,
	               const char *prefix);
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
 * @param[in] tiling: an initialized tiling.
 * @return a bitmask with order bit set (or not set) on success, an AML error
 *         (<0) on failure.
 *	   Output value can be further checked against order AML_TILING_ORDER
 *	   flags by using the macro AML_TILING_ORDER() on the output value.
 * @see AML_TILING_ORDER()
 **/
int aml_tiling_order(const struct aml_tiling *tiling);

/**
 * Return the tiling dimensions in the user order.
 * @param[in] tiling: an initialized tiling.
 * @param[out] dims: a non-NULL array of dimensions to fill. It is
 * supposed to be large enough to contain aml_tiling_ndims() elements.
 * @return AML_SUCCESS on success, else an AML error code.
 **/
int aml_tiling_dims(const struct aml_tiling *tiling, size_t *dims);

/**
 * Provide the number of dimensions in a tiling.
 * @param[in] tiling: an initialized tiling structure.
 * @return the number of dimensions in the tiling.
 **/
size_t aml_tiling_ndims(const struct aml_tiling *tiling);

/**
 * Get the dimensions of a specific tile in the tiling.
 * @param[in] tiling: the tiling to inspect.
 * @param[in] coords: the coordinate of the tile to lookup.
 * If NULL, the first tile is used.
 * @param[out] dims: the tile dimensions.
 * @return AML_SUCCESS on success.
 * @return the result of aml_tiling_index on error.
 */
int aml_tiling_tile_dims(const struct aml_tiling *tiling,
                         const size_t *coords,
                         size_t *dims);

/**
 * Provide the number of tiles in a tiling.
 * @param[in] tiling: an initialized tiling structure.
 * @return the number of tiles in the tiling.
 **/
size_t aml_tiling_ntiles(const struct aml_tiling *tiling);

/**
 * Return the tile at specified coordinates in the tiling
 * @param[in] tiling: an initialized tiling
 * @param[in] coords: the coordinates for the tile
 * @return the tile as a layout on success, NULL on error.
 **/
struct aml_layout *aml_tiling_index(const struct aml_tiling *tiling,
                                    const size_t *coords);

/**
 * Return a pointer to the first valid coordinate in the underlying tile.
 * @param[in] tiling: an initialized tiling
 * @param[in] coords: the coordinates for the tile
 * @return a raw pointer to the start of the buffer for a tile, NULL on error.
 */
void *aml_tiling_rawptr(const struct aml_tiling *tiling, const size_t *coords);

/**
 * Return the tile at the coordinates at the current position of the input
 * iterator.
 * @param[in] tiling: an initialized tiling
 * @param[in] iterator: an initialized iterator
 * @return the tile as a layout on success, NULL on error.
 */
struct aml_layout *aml_tiling_index_byiter(const struct aml_tiling *tiling,
                                           const_excit_t iterator);

/**
 * Print on the file handle the metadata associated with this tiling.
 * @param[in] stream: the stream to print on
 * @param[in] prefix: prefix to use on all lines
 * @param[in] tiling: tiling to print
 * @return 0 if successful, an error code otherwise.
 */
int aml_tiling_fprintf(FILE *stream,
                       const char *prefix,
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
 * @param[out] dst: destination layout
 * @param[in] src: source layout
 * @param[in, out] arg: extra argument needed by the operator
 **/
typedef int (*aml_dma_operator)(struct aml_layout *dst,
                                const struct aml_layout *src,
                                void *arg);

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
	 * @param[inout] dma: dma_implementation internal data.
	 * @param[out] req: the request handle to manage termination
	 *        of the movement.
	 * @param[out] dest: layout describing the destination.
	 * @param[in] src: layout describing the source.
	 * @return an AML error code.
	 **/
	int (*create_request)(struct aml_dma_data *dma,
	                      struct aml_dma_request **req,
	                      struct aml_layout *dest,
	                      struct aml_layout *src,
	                      aml_dma_operator op,
	                      void *op_arg);

	/**
	 * Destroy the request handle. If the data movement is still ongoing,
	 * then cancel it.
	 *
	 * @param[in] dma: dma_implementation internal data.
	 * @param[inout] req: the request handle to manage termination of the
	 *movement.
	 * @return an AML error code.
	 **/
	int (*destroy_request)(struct aml_dma_data *dma,
	                       struct aml_dma_request **req);

	/**
	 * Wait for termination of a data movement and destroy the request
	 * handle.
	 *
	 * @param[in] dma: dma_implementation internal data.
	 * @param[inout] req: the request handle to manage termination of the
	 *movement.
	 * @return an AML error code.
	 **/
	int (*wait_request)(struct aml_dma_data *dma,
	                    struct aml_dma_request **req);

	/**
	 * Print the implementation-specific information available on a dma.
	 * @param[in] stream: the stream to print to
	 * @param[in] prefix: a prefix string to use on all lines
	 * @param[in] data: non-NULL handle to dma internal data.
	 * @return 0 if successful, an error code otherwise.
	 **/
	int (*fprintf)(const struct aml_dma_data *data,
	               FILE *stream,
	               const char *prefix);
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
 * Request a synchronous data copy between two different buffers.
 *
 * Layouts are copied internally if necessary, avoiding the need for users to
 * keep the layouts alive during the request.
 *
 * @param[in, out] dma: an initialized DMA structure.
 * @param[out] dest: layout describing the destination.
 * @param[in] src: layout describing the source.
 * @param[in] op: optional custom operator for this dma
 * @param[in] op_arg: optional argument to the operator
 * @return 0 if successful; an error code otherwise.
 **/
int aml_dma_copy_custom(struct aml_dma *dma,
                        struct aml_layout *dest,
                        struct aml_layout *src,
                        aml_dma_operator op,
                        void *op_arg);

/**
 * Request a data copy between two different buffers.This is an asynchronous
 * version of aml_dma_copy().
 *
 * Layouts are copied internally if necessary, avoiding the need for users to
 * keep the layouts alive during the request.
 *
 * @param[in, out] dma: an initialized DMA structure.
 * @param[in, out] req: an address where the pointer to the newly assigned DMA
 *	  request will be stored.
 * @param[out] dest: layout describing the destination.
 * @param[in] src: layout describing the source.
 * @param[in] op: optional custom operator for this dma
 * @param[in] op_arg: optional argument to the operator
 * @return 0 if successful; an error code otherwise.
 **/
int aml_dma_async_copy_custom(struct aml_dma *dma,
                              struct aml_dma_request **req,
                              struct aml_layout *dest,
                              struct aml_layout *src,
                              aml_dma_operator op,
                              void *op_arg);

#define aml_dma_copy(dma, d, s) aml_dma_copy_custom(dma, d, s, NULL, NULL)
#define aml_dma_async_copy(dma, r, d, s)                                       \
	aml_dma_async_copy_custom(dma, r, d, s, NULL, NULL)

/**
 * Wait for an asynchronous DMA request to complete.
 * @param[in, out] dma: n initialized DMA structure.
 * @param[in, out] req: a DMA request obtained using aml_dma_async_*() calls.
 * @return 0 if successful; an error code otherwise.
 **/
int aml_dma_wait(struct aml_dma *dma, struct aml_dma_request **req);

/**
 * Tear down an asynchronous DMA request before it completes.
 * @param[in, out] dma: an initialized DMA structure.
 * @param[in, out] req: a DMA request obtained using aml_dma_async_*() calls.
 * @return 0 if successful; an error code otherwise.
 **/
int aml_dma_cancel(struct aml_dma *dma, struct aml_dma_request **req);

/**
 * Print on the file handle the metadata associated with this dma.
 * @param[in] stream: the stream to print on
 * @param[in] prefix: prefix to use on all lines
 * @param[in] dma: DMA to print
 * @return 0 if successful, an error code otherwise.
 */
int aml_dma_fprintf(FILE *stream,
                    const char *prefix,
                    const struct aml_dma *dma);

/**
 * Generic helper to copy from one layout to another.
 * @param[out] dst: destination layout
 * @param[in] src: source layout
 * @param[in] arg: unused (should be NULL)
 */
int aml_copy_layout_generic(struct aml_layout *dst,
                            const struct aml_layout *src,
                            void *arg);

/**
 * Helper to copy from one layout to another layout with different dimensions.
 * @param[out] dst: destination layout
 * @param[in] src: source layout
 * @param[in] target_dims: a non_NULL array with the dimensions of the
 * destination layout.
 */
int aml_copy_layout_transform_generic(struct aml_layout *dst,
                                      const struct aml_layout *src,
                                      const size_t *target_dims);

////////////////////////////////////////////////////////////////////////////////

/**
 * @}
 **/

#ifdef __cplusplus
}
#endif

#endif
