#ifndef AML_LAYOUT_RESHAPE_H
#define AML_LAYOUT_RESHAPE_H

#include <stdarg.h>

struct aml_layout_data_reshape {
	struct aml_layout *target;
	size_t ndims;
	size_t target_ndims;
	size_t *dims;
	size_t *coffsets;
	size_t *target_dims;
	size_t *target_coffsets;
};

#define AML_LAYOUT_RESHAPE_ALLOCSIZE(ndims, target_ndims) ( \
	sizeof(struct aml_layout) + \
        sizeof(struct aml_layout_data_reshape) + \
	2 * ndims * sizeof(size_t) + \
	target_ndims * sizeof(size_t) )

#define AML_LAYOUT_RESHAPE_DECL(name, ndims, target_ndims) \
	size_t __ ##name## _inner_data[ 2 * ndims + target_ndims]; \
	struct aml_layout_data_reshape __ ##name## _inner_struct = { \
		NULL, \
		ndims, \
		target_ndims, \
		__ ##name## _inner_data, \
		__ ##name## _inner_data + ndims \
		__ ##name## _inner_data + 2 * ndims \
	}; \
	struct aml_layout name = { \
		0, \
		NULL, \
		(struct aml_layout_data *)& __ ##name## _inner_struct \
	};

int aml_layout_reshape_struct_init(struct aml_layout *l, size_t ndims,
				   void *data);
int aml_layout_reshape_ainit(struct aml_layout *l, uint64_t tags,
			     struct aml_layout *target, size_t ndims,
			     const size_t *dims);
int aml_layout_reshape_vinit(struct aml_layout *l, uint64_t tags,
			     struct aml_layout *target, size_t ndims,
			     va_list data);
int aml_layout_reshape_init(struct aml_layout *l, uint64_t tags,
			    struct aml_layout *target, size_t ndims, ...);
int aml_layout_reshape_acreate(struct aml_layout **l, uint64_t tags,
			       struct aml_layout *target, size_t ndims,
			       const size_t *dims);
int aml_layout_reshape_vcreate(struct aml_layout **l, uint64_t tags,
			       struct aml_layout *target, size_t ndims,
			       va_list data);
int aml_layout_reshape_create(struct aml_layout **l, uint64_t tags,
			      struct aml_layout *target, size_t ndims, ...);

extern struct aml_layout_ops aml_layout_reshape_column_ops;
extern struct aml_layout_ops aml_layout_reshape_row_ops;

#endif
