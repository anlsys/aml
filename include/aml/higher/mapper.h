/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#ifndef AML_MAPPER_H
#define AML_MAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup aml_mapper "AML Struct Mapper"
 * @brief Hierarchical description of structs.
 *
 * Mapper a description of a data structure.
 *
 * A mapper contains the metadata to perform a complete walk or deep copy of a
 * data structure with field indirections and array fields.
 * In a mapper, a structure is described as contiguous set of bytes of some
 * size.
 * A mapper also contains metadata about some structure fields.
 * Fields are identified and located by their offset in the parent structure.
 * Fields identified in a mapper are only pointer fields that
 * represent data that cannot be copied with a simple `memcpy()` of the parent
 * structure. Pointer fields can either be a single element or an array of
 * elements. Therefore, the mapper structure also stores for each field a method
 * to obtain the number of elements stored in a field.
 * @{
 **/

/**
 * This flag denotes that the structure described should be treated
 * as an entity from different memory mapped region.
 * When walking a structure, this flags helps to compute the size of subsets
 * of a structure and allocate them in a different areas.
 */
#define AML_MAPPER_FLAG_SPLIT 0x1

/**
 * Set this flag is compound with the flag `AML_MAPPER_FLAG_SPLIT` and
 * denotes that a structure should be treated as an entity mapped on host with
 * `malloc()`.
 * This is typically used to deepcopy a structure that has its top level mapped
 * on host while some fields are mapped in a different device memory.
 */
#define AML_MAPPER_FLAG_HOST 0x3

typedef size_t (*num_element_fn)(void *);

struct aml_mapper {
	// OR combination of AML_MAPPER_FLAG_*
	uint64_t flags;
	// The top level struct size (`sizeof()`).
	size_t size;
	// The number of pointer fields in the struct that this mapper
	// describes.
	size_t n_fields;
	// The offset of each field in the top level struct.
	size_t *offsets;
	// An array of function pointers. Each function takes as input a pointer
	// to a struct represented by this mapper and returns the number of
	// elements in corresponding field. Any field with NULL function pointer
	// is considered to have a single element. If the array itself is NULL,
	// then all fields are considered to be single elements.
	num_element_fn *num_elements;
	// The mapper of child fields.
	struct aml_mapper **fields;
};

//-----------------------------------------------------------------------------
// Constructor / Destructor
//-----------------------------------------------------------------------------

/**
 * Dynamic constructor for a struct mapper constructor.
 *
 * In most cases, a mapper can be instanciated statically with the macros:
 * `aml_mapper_decl()` and `aml_final_mapper_decl()`.
 *
 * @param[out] out: A pointer to where mapper should be allocated.
 * @param[in] flags: A ORed set of `AML_MAPPER_FLAG_*`.
 * @param[in] num_fields: The number of fields that are pointers
 * to cross in order to map the complete structure.
 * @param[in] fields_offset: The offset of each field in the structure.
 * @param[in] num_elements: An array of function pointers returning the number
 * of element in each field. Any NULL function pointer is considered to have
 * one child pointer field to cross. If the array itself is NULL, then all
 * fields are considered to be single elements.
 * @param[in] fields: An array of mapper for each pointer field to map.
 * @return AML_SUCCESS on success.
 * @return -AML_ENOMEM if allocation failed.
 * @return -AML_EINVAL if `num_fields` is not 0 and `fields_offset` or
 * `fields` is NULL.
 */
int aml_mapper_create(struct aml_mapper **out,
                      uint64_t flags,
                      const size_t struct_size,
                      const size_t num_fields,
                      const size_t *fields_offset,
                      const num_element_fn *num_elements,
                      struct aml_mapper **fields);

/**
 * Struct mapper destructor.
 * @param[in/out] mapper: A pointer to the mapper to delete.
 * The content of mapper is set to NULL afterward.
 */
void aml_mapper_destroy(struct aml_mapper **mapper);

/**
 * Declare a static mapper for a structure type.
 * @param[in] name: The result mapper variable name.
 * @param[in] flags: A ORed combination of flags: `AML_MAPPER_FLAG_*`.
 * @param[in] type: The type of the structure to map.
 * @param[in] __VA_ARGS__: Must contain a multiple of 2 or 3 arguments.
 * If empty, then the structure is considered plain, it will be mapped by
 * mapper but none of its field will be descended when mapping.
 * If a multiple of 2, it must be a list of (`field`, `field_mapper`) where
 *   - `field` is the name of a field to map in `type` struct,
 *   - `field_mapper`: A pointer to a `struct aml_mapper` that maps
 * this field type.
 * If a multiple of 3, it must be a list of
 * (`field`, `num_elements`, `field_mapper`) where
 *   - `field` is the name of a field to map in `type` struct,
 *   - `num_elements` is the name of the struct field that contains the number
 * of struct `field` contiguous elements pointed by struct field `field`.
 * The type of the field that counts the number of elements must be a
 * `size_t`. If it is a different size or different type, the behaviour of
 * using the resulting mapper is undefined.
 *   - `field_mapper` is a pointer to a `struct aml_mapper` that maps
 * this field type.
 **/
#define aml_mapper_decl(name, flags, type, ...)                                \
	CONCATENATE(__AML_MAPPER_DECL_, __AML_MAPPER_DECL_SELECT(__VA_ARGS__)) \
	(name, flags, type, __VA_ARGS__)

/**
 * Declare a static mapper for a structure type that does not need
 * to be descended in the copy.
 * @param[in] name: The result mapper variable name.
 * @param[in] flags: A ORed combination of flags: `AML_MAPPER_FLAG_*`.
 * @param[in] type: The type of structure to map.
 **/
#define aml_final_mapper_decl(name, flags, type)                               \
	struct aml_mapper name =                                               \
	        __AML_MAPPER_INIT(flags, type, 0, NULL, NULL, NULL)

//-----------------------------------------------------------------------------
// Default Mappers
//-----------------------------------------------------------------------------

/** Default mapper for elements of type char */
extern struct aml_mapper aml_char_mapper;
/** Default mapper for elements of type short */
extern struct aml_mapper aml_short_mapper;
/** Default mapper for elements of type int */
extern struct aml_mapper aml_int_mapper;
/** Default mapper for elements of type long */
extern struct aml_mapper aml_long_mapper;
/** Default mapper for elements of type long long */
extern struct aml_mapper aml_long_long_mapper;
/** Default mapper for elements of type unsigned char */
extern struct aml_mapper aml_uchar_mapper;
/** Default mapper for elements of type unsigned int */
extern struct aml_mapper aml_uint_mapper;
/** Default mapper for elements of type unsigned long */
extern struct aml_mapper aml_ulong_mapper;
/** Default mapper for elements of type unsigned long long */
extern struct aml_mapper aml_ulong_long_mapper;
/** Default mapper for elements of type float */
extern struct aml_mapper aml_float_mapper;
/** Default mapper for elements of type double */
extern struct aml_mapper aml_double_mapper;
/** Default mapper for elements of type long double */
extern struct aml_mapper aml_long_double_mapper;
/** Default mapper for pointer elements */
extern struct aml_mapper aml_ptr_mapper;
/** Default mapper in a seperate allocation for elements of type char
 */
extern struct aml_mapper aml_char_split_mapper;
/** Default mapper in a seperate allocation for elements of type short
 */
extern struct aml_mapper aml_short_split_mapper;
/** Default mapper in a seperate allocation for elements of type int
 */
extern struct aml_mapper aml_int_split_mapper;
/** Default mapper in a seperate allocation for elements of type long
 */
extern struct aml_mapper aml_long_split_mapper;
/** Default mapper in a seperate allocation for elements of type long
 * long */
extern struct aml_mapper aml_long_long_split_mapper;
/** Default mapper in a seperate allocation for elements of type
 * unsigned char */
extern struct aml_mapper aml_uchar_split_mapper;
/** Default mapper in a seperate allocation for elements of type
 * unsigned short */
extern struct aml_mapper aml_ushort_split_mapper;
/** Default mapper in a seperate allocation for elements of type
 * unsigned int */
extern struct aml_mapper aml_uint_split_mapper;
/** Default mapper in a seperate allocation for elements of type
 * unsigned long */
extern struct aml_mapper aml_ulong_split_mapper;
/** Default mapper in a seperate allocation for elements of type
 * unsigned long long */
extern struct aml_mapper aml_ulong_long_split_mapper;
/** Default mapper in a seperate allocation for elements of type float
 */
extern struct aml_mapper aml_float_split_mapper;
/** Default mapper in a seperate allocation for elements of type
 * double */
extern struct aml_mapper aml_double_split_mapper;
/** Default mapper in a seperate allocation for elements of type long
 * double */
extern struct aml_mapper aml_long_double_split_mapper;
/** Default mapper in a seperate allocation for pointer elements */
extern struct aml_mapper aml_ptr_split_mapper;

/**
 * @}
 **/

#ifdef __cplusplus
}
#endif
#endif // AML_MAPPER_H
