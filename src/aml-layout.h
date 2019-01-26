#ifndef AML_LAYOUT_H
#define AML_LAYOUT_H 1

#include <stdarg.h>

/*******************************************************************************
 * Data Layout Management:
 ******************************************************************************/

struct aml_layout;
struct aml_layout_data;

/*******************************************************************************
 * Generic layout, with support for sparsity and strides.
 ******************************************************************************/

/* Layout type tags. Defined as the bit offset to set to one. */
#define AML_TYPE_LAYOUT_ORDER (1 << 0)
#define AML_TYPE_LAYOUT_MAX (1 << 1)

#define AML_TYPE_LAYOUT_ROW_ORDER 1
#define AML_TYPE_LAYOUT_COLUMN_ORDER 0

#define AML_TYPE_GET(tags, bit) (tags & bit)
#define AML_TYPE_CLEAR(tags, bit) (tags &= ~bit)
#define AML_TYPE_SET(tags, bit, value) do { \
	AML_TYPE_CLEAR(tags, bit); \
	if(value) tags |= bit;} while(0)


struct aml_layout_ops {
	void *(*deref)(const struct aml_layout_data *, va_list coords);
	void *(*aderef)(const struct aml_layout_data *, const size_t *coords);
	int (*order)(const struct aml_layout_data *);
	int (*dims)(const struct aml_layout_data *, va_list dim_ptrs);
	int (*adims)(const struct aml_layout_data *, size_t *dims);
	int (*adims_column)(const struct aml_layout_data *, size_t *dims);
        size_t (*ndims)(const struct aml_layout_data *);
        size_t (*element_size)(const struct aml_layout_data *);
        struct aml_layout * (*reshape)(const struct aml_layout_data *,
				       size_t ndims, va_list dims);
        struct aml_layout * (*areshape)(const struct aml_layout_data *,
					size_t ndims, const size_t *dims);
};

struct aml_layout {
	uint64_t tags;
	struct aml_layout_ops *ops;
	struct aml_layout_data *data;
};

void *aml_layout_deref(const struct aml_layout *l, ...);
void *aml_layout_aderef(const struct aml_layout *l, const size_t *coords);
int aml_layout_order(const struct aml_layout *l);
int aml_layout_dims(const struct aml_layout *l, ...);
int aml_layout_adims(const struct aml_layout *l, size_t *dims);
size_t aml_layout_ndims(const struct aml_layout *l);
size_t aml_layout_element_size(const struct aml_layout *l);
struct aml_layout * aml_layout_areshape(const struct aml_layout *l,
					size_t ndims, const size_t *dims);
struct aml_layout * aml_layout_reshape(const struct aml_layout *l,
				       size_t ndims, ...);

#endif
