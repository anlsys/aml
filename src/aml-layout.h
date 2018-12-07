#ifndef AML_LAYOUT_H
#define AML_LAYOUT_H 1

/*******************************************************************************
 * Data Layout Management:
 ******************************************************************************/

struct aml_layout;
struct aml_layout_data;

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

#define AML_TYPE_LAYOUT_COLUMN_ORDER 0
#define AML_TYPE_LAYOUT_ROW_ORDER 1

struct aml_layout_data {
	void *ptr;
	size_t ndims;
	size_t *dims;
	size_t *pitch;
	size_t *stride;
};

struct aml_layout_ops {
	void *(*deref)(const struct aml_layout_data *, va_list coords);
	void *(*aderef)(const struct aml_layout_data *, const size_t *coords);
	int (*order)(const struct aml_layout_data *);
	int (*dims)(const struct aml_layout_data *, va_list dim_ptrs);
	int (*adims)(const struct aml_layout_data *, const size_t *dims);
};

struct aml_layout {
	uint64_t tags;
	struct aml_layout_ops *ops;
	struct aml_layout_data *data;
};

#define AML_LAYOUT_ALLOCSIZE(ndims) (sizeof(struct aml_layout) +\
					sizeof(struct aml_layout_data) +\
					ndims * 3 * sizeof(size_t))

#define AML_LAYOUT_DECL(name, ndims) \
	size_t __ ##name## _inner_data[ndims * 3]; \
	struct aml_layout_data __ ##name## _inner_struct = { \
		NULL, \
		ndims, \
		__ ##name## _inner_data, \
		__ ##name## _inner_data + ndims, \
		__ ##name## _inner_data + 2 * ndims, \
	}; \
	struct aml_layout name = { \
		0, \
		NULL, \
		& __ ##name## _inner_struct, \
	};

int aml_layout_struct_init(struct aml_layout *l, size_t ndims, void *data);
int aml_layout_ainit(struct aml_layout *l, uint64_t tags, void *ptr,
		     const size_t element_size, size_t ndims,
		    const size_t *dims, const size_t *stride,
		    const size_t *pitch);
int aml_layout_vinit(struct aml_layout *l, uint64_t tags, void *ptr,
		     const size_t element_size, size_t ndims, va_list data);
int aml_layout_init(struct aml_layout *l, uint64_t tags, void *ptr,
		     const size_t element_size, size_t ndims, ...);
int aml_layout_acreate(struct aml_layout **l, uint64_t tags, void *ptr,
		      const size_t element_size, size_t ndims,
		      const size_t *dims, const size_t *stride,
		      const size_t *pitch);
int aml_layout_vcreate(struct aml_layout **l, uint64_t tags, void *ptr,
		      const size_t element_size, size_t ndims, va_list data);
int aml_layout_create(struct aml_layout **l, uint64_t tags, void *ptr,
		      const size_t element_size, size_t ndims, ...);

void *aml_layout_deref(const struct aml_layout *l, ...);
int aml_layout_order(const struct aml_layout *l);
int aml_layout_dims(const struct aml_layout *l, ...);

/*******************************************************************************
 * Dense Layout Operators.
 ******************************************************************************/

void *aml_layout_column_deref(const struct aml_layout_data *d, va_list coords);
void *aml_layout_column_aderef(const struct aml_layout_data *d, size_t *coords);
int aml_layout_column_order(const struct aml_layout_data *d);
int aml_layout_column_dims(const struct aml_layout_data *d, va_list dims);
int aml_layout_column_adims(const struct aml_layout_data *d, const size_t *dims);

extern struct aml_layout_ops aml_layout_column_ops;

void *aml_layout_row_deref(const struct aml_layout_data *d, va_list coords);
void *aml_layout_row_aderef(const struct aml_layout_data *d, size_t *coords);
int aml_layout_row_order(const struct aml_layout_data *d);
int aml_layout_row_dims(const struct aml_layout_data *d, va_list dims);
int aml_layout_row_adims(const struct aml_layout_data *d, const size_t *dims);

extern struct aml_layout_ops aml_layout_row_ops;
#endif
