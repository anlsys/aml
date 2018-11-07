#ifndef AML_LAYOUT_H
#define AML_LAYOUT_H 1

/*******************************************************************************
 * Data Layout Management:
 ******************************************************************************/

struct aml_layout;

/*******************************************************************************
 * Generic layout, with support for sparsity and strides.
 ******************************************************************************/

/* Layout: describes how a  multi-dimensional data structure is collapsed into a
 * linear (and contiguous) virtual address range.
 * "ptr": base pointer of the address range
 * "ndims": number of dimensions
 * "dims": dimensions, in element size, of the data structure, by order of
 *         appearance in memory.
 * "pitch": cumulative distances between two elements in the same dimension
 *          (pitch[0] is the element size in bytes).
 * "stride": offset between elements of the same dimension.
 */

struct aml_layout {
	void *ptr;
	size_t ndims;
	size_t *dims;
	size_t *pitch;
	size_t *stride;
};

#define AML_LAYOUT_ALLOCSIZE(ndims) (sizeof(struct aml_layout) +\
					    ndims * 3 * sizeof(size_t))

#define AML_LAYOUT_DECL(name, ndims) \
	size_t __ ##name## _inner_data[ndims * 3]; \
	struct aml_layout name = { \
		NULL, \
		ndims, \
		__ ##name## _inner_data, \
		__ ##name## _inner_data + ndims, \
		__ ##name## _inner_data + 2 * ndims, \
	};

int aml_layout_struct_init(struct aml_layout *l, size_t ndims, void *data);
int aml_layout_init(struct aml_layout *l, void *ptr, size_t ndims,
		    const size_t *dims, const size_t *pitch,
		    const size_t *stride);
int aml_layout_create(struct aml_layout **l, void *ptr, size_t ndims,
		    const size_t *dims, const size_t *pitch,
		    const size_t *stride);

#endif
