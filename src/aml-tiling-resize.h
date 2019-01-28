#ifndef AML_TILING_RESIZE_H
#define AML_TILING_RESIZE_H

#include <stdarg.h>

struct aml_tiling_nd_data_resize {
	struct aml_layout *l;
	size_t ndims;
	size_t *tile_dims;
	size_t *dims;
	size_t *border_tile_dims;
};

#define AML_TILING_RESIZE_ALLOCSIZE(ndims) (sizeof(struct aml_tiling_nd) +\
					    sizeof(struct aml_tiling_nd_data_resize) +\
					    (ndims * 3) * sizeof(size_t))

int aml_tiling_nd_resize_struct_init(struct aml_tiling_nd *t, size_t ndims,
				     void *data);
int aml_tiling_nd_resize_ainit(struct aml_tiling_nd *t, uint64_t tags,
			       const struct aml_layout *l, size_t ndims,
			       const size_t *tile_dims);
int aml_tiling_nd_resize_vinit(struct aml_tiling_nd *t, uint64_t tags,
			       const struct aml_layout *l, size_t ndims,
			       va_list data);
int aml_tiling_nd_resize_init(struct aml_tiling_nd *t, uint64_t tags,
			      const struct aml_layout *l, size_t ndims, ...);
int aml_tiling_nd_resize_acreate(struct aml_tiling_nd **t, uint64_t tags,
				 const struct aml_layout *l, size_t ndims,
				 const size_t *tile_dims);
int aml_tiling_nd_resize_vcreate(struct aml_tiling_nd **t, uint64_t tags,
				 const struct aml_layout *l, size_t ndims,
				 va_list data);
int aml_tiling_nd_resize_create(struct aml_tiling_nd **t, uint64_t tags,
				const struct aml_layout *l, size_t ndims, ...);

extern struct aml_tiling_nd_ops aml_tiling_nd_resize_column_ops;
extern struct aml_tiling_nd_ops aml_tiling_nd_resize_row_ops;

#endif
