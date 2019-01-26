#ifndef AML_TILING_H
#define AML_TILING_H 1

#include <stdarg.h>

struct aml_tiling_nd;
struct aml_tiling_nd_data;

#define AML_TYPE_TILING_ORDER (1 << 0)
#define AML_TYPE_TILING_MAX (1 << 1)

#define AML_TYPE_LILING_ROW_ORDER 1
#define AML_TYPE_LILING_COLUMN_ORDER 0

struct aml_tiling_nd_ops {
	struct aml_layout* (*index)(const struct aml_tiling_nd_data *,
				    va_list coords);
	struct aml_layout* (*aindex)(const struct aml_tiling_nd_data *,
				     const size_t *coords);
	int (*order)(const struct aml_tiling_nd_data *);
        int (*tile_dims)(const struct aml_tiling_nd_data *, va_list dim_ptrs);
	int (*tile_adims)(const struct aml_tiling_nd_data *, size_t *dims);
	int (*dims)(const struct aml_tiling_nd_data *, va_list dim_ptrs);
	int (*adims)(const struct aml_tiling_nd_data *, size_t *dims);
	size_t (*ndims)(const struct aml_tiling_nd_data *);
};

struct aml_tiling_nd {
	uint64_t tags;
	struct aml_tiling_nd_ops *ops;
	struct aml_tiling_nd_data *data;
};

struct aml_layout *aml_tiling_nd_index(const struct aml_tiling_nd *t, ...);
struct aml_layout *aml_tiling_nd_aindex(const struct aml_tiling_nd *t,
					const size_t *coords);
int aml_tiling_nd_order(const struct aml_tiling_nd *t);
int aml_tiling_nd_tile_dims(const struct aml_tiling_nd *t, ...);
int aml_tiling_nd_tile_adims(const struct aml_tiling_nd *t, size_t *dims);
int aml_tiling_nd_dims(const struct aml_tiling_nd *t, ...);
int aml_tiling_nd_adims(const struct aml_tiling_nd *t, size_t *dims);
size_t aml_tiling_nd_ndims(const struct aml_tiling_nd *t);

#endif
