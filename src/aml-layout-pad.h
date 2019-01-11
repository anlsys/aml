#ifndef AML_LAYOUT_PAD_H
#define AML_LAYOUT_PAD_H 1

#include <stdarg.h>

struct aml_layout_data_pad {
	struct aml_layout *target;
	size_t ndims;
	size_t element_size;
	size_t *dims;
	size_t *target_dims;
	void *neutral;
};


#define AML_LAYOUT_PAD_ALLOCSIZE(ndims, neutral_size) ( \
	sizeof(struct aml_layout) + \
	sizeof(struct aml_layout_data_pad) + \
	2 * ndims * sizeof(size_t) + \
	neutral_size )

#define AML_LAYOUT_PAD_DECL(name, ndims, neutral_size) \
	uint8_t __ ##name## _inner_data[ndims * sizeof(size_t) + \
					neutral_size ]; \
	struct aml_layout_data_pad __ ##name## _inner_struct = { \
		NULL, \
		ndims, \
		neutral_size, \
		(size_t *) __ ##name## _inner_data, \
		(size_t *) __ ##name## _inner_data + ndims * sizeof(size_t), \
		(void *) __ ##name## _inner_data + 2 * ndims * sizeof(size_t) \
	}; \
	struct aml_layout name = { \
		0, \
		NULL, \
		(struct aml_layout_data *)& __ ##name## _inner_struct \
	};

int aml_layout_pad_struct_init(struct aml_layout *l, size_t ndims,
			       size_t element_size, void *data);
int aml_layout_pad_ainit(struct aml_layout *l, uint64_t tags,
			 struct aml_layout *target, const size_t *dims,
			 void *neutral);
int aml_layout_pad_vinit(struct aml_layout *l, uint64_t tags,
			 struct aml_layout *target, va_list data);
int aml_layout_pad_init(struct aml_layout *l, uint64_t tags,
			struct aml_layout *target, ...);
int aml_layout_pad_acreate(struct aml_layout **l, uint64_t tags,
			   struct aml_layout *target, const size_t *dims,
			   void *neutral);
int aml_layout_pad_vcreate(struct aml_layout **l, uint64_t tags,
			   struct aml_layout *target, va_list data);
int aml_layout_pad_create(struct aml_layout **l, uint64_t tags,
			  struct aml_layout *target, ...);

extern struct aml_layout_ops aml_layout_pad_column_ops;
extern struct aml_layout_ops aml_layout_pad_row_ops;
#endif
