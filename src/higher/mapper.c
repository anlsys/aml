/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include "aml.h"

#include "aml/higher/mapper.h"
#include "aml/utils/inner-malloc.h"

int aml_mapper_create(struct aml_mapper **out,
                      uint64_t flags,
                      const size_t struct_size,
                      const size_t num_fields,
                      const size_t *fields_offset,
                      const num_element_fn *num_elements,
                      struct aml_mapper **fields)
{
	if (num_fields > 0 && (fields_offset == NULL || fields == NULL))
		return -AML_EINVAL;

	struct aml_mapper *m;
	size_t extra = num_fields * sizeof(struct aml_mapper *) +
	               (num_elements ? num_fields * sizeof(num_element_fn) : 0);

	if (num_fields == 0) {
		m = AML_INNER_MALLOC(struct aml_mapper);
		if (m == NULL)
			return -AML_ENOMEM;
		m->size = struct_size;
		m->n_fields = 0;
		m->offsets = NULL;
		m->num_elements = NULL;
		m->fields = NULL;
	} else {
		m = AML_INNER_MALLOC_EXTRA(num_fields, size_t, extra,
		                           struct aml_mapper);
		if (m == NULL)
			return -AML_ENOMEM;

		m->size = struct_size;
		m->n_fields = num_fields;
		m->offsets = AML_INNER_MALLOC_GET_ARRAY(m, size_t,
		                                        struct aml_mapper);
		memcpy(m->offsets, fields_offset,
		       num_fields * sizeof(*fields_offset));

		m->fields = AML_INNER_MALLOC_GET_EXTRA(m, num_fields, size_t,
		                                       struct aml_mapper);
		memcpy(m->fields, fields, num_fields * sizeof(*fields));

		if (num_elements == NULL)
			m->num_elements = NULL;
		else {
			m->num_elements =
			        (num_element_fn *)((char *)m->fields +
			                           num_fields *
			                                   sizeof(*fields));
			memcpy(m->num_elements, num_elements,
			       num_fields * sizeof(*num_elements));
		}
	}
	m->flags = flags;
	*out = m;
	return AML_SUCCESS;
}

void aml_mapper_destroy(struct aml_mapper **mapper)
{
	if (mapper) {
		if (*mapper)
			free(*mapper);
		*mapper = NULL;
	}
}

aml_final_mapper_decl(aml_char_mapper, 0, char);
aml_final_mapper_decl(aml_short_mapper, 0, short);
aml_final_mapper_decl(aml_int_mapper, 0, int);
aml_final_mapper_decl(aml_long_mapper, 0, long);
aml_final_mapper_decl(aml_long_long_mapper, 0, long long);
aml_final_mapper_decl(aml_uchar_mapper, 0, unsigned char);
aml_final_mapper_decl(aml_uint_mapper, 0, unsigned int);
aml_final_mapper_decl(aml_ulong_mapper, 0, unsigned long);
aml_final_mapper_decl(aml_ulong_long_mapper, 0, unsigned long long);
aml_final_mapper_decl(aml_float_mapper, 0, float);
aml_final_mapper_decl(aml_double_mapper, 0, double);
aml_final_mapper_decl(aml_long_double_mapper, 0, long double);
aml_final_mapper_decl(aml_ptr_mapper, 0, void *);
aml_final_mapper_decl(aml_char_split_mapper, AML_MAPPER_FLAG_SPLIT, char);
aml_final_mapper_decl(aml_short_split_mapper, AML_MAPPER_FLAG_SPLIT, short);
aml_final_mapper_decl(aml_int_split_mapper, AML_MAPPER_FLAG_SPLIT, int);
aml_final_mapper_decl(aml_long_split_mapper, AML_MAPPER_FLAG_SPLIT, long);
aml_final_mapper_decl(aml_long_long_split_mapper,
                      AML_MAPPER_FLAG_SPLIT,
                      long long);
aml_final_mapper_decl(aml_uchar_split_mapper,
                      AML_MAPPER_FLAG_SPLIT,
                      unsigned char);
aml_final_mapper_decl(aml_ushort_split_mapper,
                      AML_MAPPER_FLAG_SPLIT,
                      unsigned short);
aml_final_mapper_decl(aml_uint_split_mapper,
                      AML_MAPPER_FLAG_SPLIT,
                      unsigned int);
aml_final_mapper_decl(aml_ulong_split_mapper,
                      AML_MAPPER_FLAG_SPLIT,
                      unsigned long);
aml_final_mapper_decl(aml_ulong_long_split_mapper,
                      AML_MAPPER_FLAG_SPLIT,
                      unsigned long long);
aml_final_mapper_decl(aml_float_split_mapper, AML_MAPPER_FLAG_SPLIT, float);
aml_final_mapper_decl(aml_double_split_mapper, AML_MAPPER_FLAG_SPLIT, double);
aml_final_mapper_decl(aml_long_double_split_mapper,
                      AML_MAPPER_FLAG_SPLIT,
                      long double);
aml_final_mapper_decl(aml_ptr_split_mapper, AML_MAPPER_FLAG_SPLIT, void *);
