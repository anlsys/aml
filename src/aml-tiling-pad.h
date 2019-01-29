#ifndef AML_TILING_PAD_H
#define AML_TILING_PAD_H

#include <stdarg.h>

struct aml_tiling_nd_data_pad {
	const struct aml_layout *l;
	size_t ndims;
	size_t *tile_dims;
	size_t *dims;
	size_t *border_tile_dims;
	size_t *pad;
	void *neutral;
};

#define AML_TILING_PAD_ALLOCSIZE(ndims, neutral_size) ( \
	sizeof(struct aml_tiling_nd) + \
	sizeof(struct aml_tiling_nd_data_pad) + \
	(ndims * 4) * sizeof(size_t) + \
	neutral_size )

int aml_tiling_nd_pad_struct_init(struct aml_tiling_nd *t, size_t ndims,
				     void *data);
int aml_tiling_nd_pad_ainit(struct aml_tiling_nd *t, uint64_t tags,
			       const struct aml_layout *l, size_t ndims,
			       const size_t *tile_dims, void *neutral);
int aml_tiling_nd_pad_vinit(struct aml_tiling_nd *t, uint64_t tags,
			       const struct aml_layout *l, size_t ndims,
			       va_list data);
int aml_tiling_nd_pad_init(struct aml_tiling_nd *t, uint64_t tags,
			      const struct aml_layout *l, size_t ndims, ...);
int aml_tiling_nd_pad_acreate(struct aml_tiling_nd **t, uint64_t tags,
				 const struct aml_layout *l, size_t ndims,
				 const size_t *tile_dims, void *neutral);
int aml_tiling_nd_pad_vcreate(struct aml_tiling_nd **t, uint64_t tags,
				 const struct aml_layout *l, size_t ndims,
				 va_list data);
int aml_tiling_nd_pad_create(struct aml_tiling_nd **t, uint64_t tags,
				const struct aml_layout *l, size_t ndims, ...);

extern struct aml_tiling_nd_ops aml_tiling_nd_pad_column_ops;
extern struct aml_tiling_nd_ops aml_tiling_nd_pad_row_ops;

#endif
