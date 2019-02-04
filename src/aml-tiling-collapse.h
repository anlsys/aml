#ifndef AML_TILING_COLLAPSE_H
#define AML_TILING_COLLAPSE_H

#include <stdarg.h>

struct aml_tiling_nd_data_collapse {
	const struct aml_layout *l;
	size_t ndims;
	size_t *tile_dims;
	size_t *dims;
	size_t *border_tile_dims;
};

#define AML_TILING_COLLAPSE_ALLOCSIZE(ndims) (sizeof(struct aml_tiling_nd) +\
					    sizeof(struct aml_tiling_nd_data_collapse) +\
					    (ndims * 3) * sizeof(size_t))

int aml_tiling_nd_collapse_struct_init(struct aml_tiling_nd *t, size_t ndims,
				     void *data);
int aml_tiling_nd_collapse_ainit(struct aml_tiling_nd *t, uint64_t tags,
			       const struct aml_layout *l, size_t ndims,
			       const size_t *tile_dims);
int aml_tiling_nd_collapse_vinit(struct aml_tiling_nd *t, uint64_t tags,
			       const struct aml_layout *l, size_t ndims,
			       va_list data);
int aml_tiling_nd_collapse_init(struct aml_tiling_nd *t, uint64_t tags,
			      const struct aml_layout *l, size_t ndims, ...);
int aml_tiling_nd_collapse_acreate(struct aml_tiling_nd **t, uint64_t tags,
				 const struct aml_layout *l, size_t ndims,
				 const size_t *tile_dims);
int aml_tiling_nd_collapse_vcreate(struct aml_tiling_nd **t, uint64_t tags,
				 const struct aml_layout *l, size_t ndims,
				 va_list data);
int aml_tiling_nd_collapse_create(struct aml_tiling_nd **t, uint64_t tags,
				const struct aml_layout *l, size_t ndims, ...);

extern struct aml_tiling_nd_ops aml_tiling_nd_collapse_column_ops;
extern struct aml_tiling_nd_ops aml_tiling_nd_collapse_row_ops;

#endif
