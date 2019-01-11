#ifndef AML_LAYOUT_DENSE_H
#define AML_LAYOUT_DENSE_H 1

#include <stdarg.h>

/*******************************************************************************
 * Native Layout Operators.
 ******************************************************************************/

/* Layout: describes how a  multi-dimensional dense data structure is collapsed
 * into a linear (and contiguous) virtual address range.
 * "ptr": base pointer of the address range
 * "ndims": number of dimensions
 * "dims": dimensions, in element size, of the data structure, by order of
 *         appearance in memory.
 * "stride": offset between elements of the same dimension.
 * "pitch": distances between two elements of the next dimension (or total
            dimension of the layout in this dimension).
 * "cpitch": cumulative distances between two elements in the same dimension
 *           (pitch[0] is the element size in bytes).
 */
struct aml_layout_data_native {
	void *ptr;
	size_t ndims;
	size_t *dims;
	size_t *stride;
	size_t *pitch;
	size_t *cpitch;
};

#define AML_LAYOUT_NATIVE_ALLOCSIZE(ndims) (sizeof(struct aml_layout) +\
					sizeof(struct aml_layout_data_native) +\
					ndims * 4 * sizeof(size_t))

#define AML_LAYOUT_NATIVE_DECL(name, ndims) \
	size_t __ ##name## _inner_data[ndims * 4]; \
	struct aml_layout_data_native __ ##name## _inner_struct = { \
		NULL, \
		ndims, \
		__ ##name## _inner_data, \
		__ ##name## _inner_data + ndims, \
		__ ##name## _inner_data + 2 * ndims, \
		__ ##name## _inner_data + 3 * ndims, \
	}; \
	struct aml_layout name = { \
		0, \
		NULL, \
		(struct aml_layout_data *)& __ ##name## _inner_struct, \
	};

int aml_layout_native_struct_init(struct aml_layout *l, size_t ndims,
				  void *data);
int aml_layout_native_ainit(struct aml_layout *l, uint64_t tags, void *ptr,
			    const size_t element_size, size_t ndims,
			    const size_t *dims, const size_t *stride,
			    const size_t *pitch);
int aml_layout_native_vinit(struct aml_layout *l, uint64_t tags, void *ptr,
			    const size_t element_size, size_t ndims,
			    va_list data);
int aml_layout_native_init(struct aml_layout *l, uint64_t tags, void *ptr,
			   const size_t element_size, size_t ndims, ...);
int aml_layout_native_acreate(struct aml_layout **l, uint64_t tags, void *ptr,
			      const size_t element_size, size_t ndims,
			      const size_t *dims, const size_t *stride,
			      const size_t *pitch);
int aml_layout_native_vcreate(struct aml_layout **l, uint64_t tags, void *ptr,
			      const size_t element_size, size_t ndims,
			      va_list data);
int aml_layout_native_create(struct aml_layout **l, uint64_t tags, void *ptr,
			     const size_t element_size, size_t ndims, ...);

void *aml_layout_column_deref(const struct aml_layout_data *data,
			      va_list coords);
void *aml_layout_column_aderef(const struct aml_layout_data *data,
			       const size_t *coords);
int aml_layout_column_order(const struct aml_layout_data *data);
int aml_layout_column_dims(const struct aml_layout_data *data, va_list dims);
int aml_layout_column_adims(const struct aml_layout_data *data, size_t *dims);
size_t aml_layout_column_ndims(const struct aml_layout_data *data);
size_t aml_layout_column_elem_size(const struct aml_layout_data *data);

extern struct aml_layout_ops aml_layout_column_ops;

void *aml_layout_row_deref(const struct aml_layout_data *data, va_list coords);
void *aml_layout_row_aderef(const struct aml_layout_data *data,
			    const size_t *coords);
int aml_layout_row_order(const struct aml_layout_data *data);
int aml_layout_row_dims(const struct aml_layout_data *data, va_list dims);
int aml_layout_row_adims(const struct aml_layout_data *data, size_t *dims);
size_t aml_layout_row_ndims(const struct aml_layout_data *data);
size_t aml_layout_row_element_size(const struct aml_layout_data *data);

extern struct aml_layout_ops aml_layout_row_ops;

#endif
