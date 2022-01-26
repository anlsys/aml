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
 * A mapper should be able to be instanciated once per `struct` declaration
 * and accurately describe any dynamic instance of the same structure.
 * Mapper can then be used to deep-copy complex structures in different memory
 * regions.
 * @{
 **/

/**
 * When walking a user structure, the expected behavior is to walk the entire
 * structure. When this flag is set, the walk should stop when encountering this
 * mapper and take the appropriate action for the structure described and its
 * offsprings. For instance when performing a deep copy, the copy of this
 * structure may be done in a different buffer compared to the parent structure,
 * or when computing the total size of a parent structure, the size of this
 * structure could be omitted.
 */
#define AML_MAPPER_FLAG_SPLIT 0x1

/**
 * The default mapper behaviour is to allocate a new piece of memory for the
 * top level structure of a mapped structure hierarchy.
 * When this flag is set, the `dst` pointer to map is assumed to be already
 * allocated ON HOST, and only its field are to be allocated. This flag can
 * only be used on the top level structure/mapper and connected descendant
 * holding the same flag. If this flag is set in any mapper of a mapper
 * hierarchy that is not connected to a mapper with the same flag, up to the
 * root, then it is silently ignored.
 */
#define AML_MAPPER_FLAG_SHALLOW 0x2

typedef size_t (*num_element_fn)(void *);

struct aml_mapper {
	// OR combination of AML_MAPPER_FLAG_*
	uint64_t flags;
	// The struct size.
	size_t size;
	// The number of pointer fields in the struct
	// that are to be crossed.
	size_t n_fields;
	// The offset of each field in struct.
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
 * Struct mapper constructor.
 * User of this function must be careful as it may result in undefined
 *  behaviour:
 * If the mapper structure `out` is not a tree, e.g it has mapper fields of the
 * same type as it self and mappers are pointing at each others, then the
 * behaviour of this function is undefined.
 * This function does not implement cycles detection and
 * will loop on cycles, dereferencing data out of memory bounds and eventually
 * loop allocating all memory if a cycle is met.
 * @param[out] out: A pointer to where mapper should be allocated.
 * @param[in] flags: A ORed set of `AML_MAPPER_FLAG_*` to customize mapper
 * behaviour.
 * @see `AML_MAPPER_FLAG_SPLIT`
 * @see `AML_MAPPER_FLAG_SHALLOW`
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
 * Declare a static mapper for a structure type with custom flags.
 * @param[in] name: The result mapper variable name.
 * @param[in] flags: A or combination of flags: `AML_MAPPER_FLAG_*`.
 * @param[in] type: The type of structure to map.
 * @param[in] __VA_ARGS__: Must contain a multiple of 2 or 3 arguments.
 * If empty, then the structure is considered plain, it will be mapped by
 * mapper but none of its field will be descended when mapping.
 * If a multiple of 2, must be a list of (field, field_mapper) where
 *   - field: The name of a field to map in `type` struct;
 *   - field_mapper: A pointer to a `struct aml_mapper` that maps
 * this field type.
 * If a multiple of 2, must be a list of (field, num_elements, field_mapper)
 * where
 *   - field: The name of a field to map in `type` struct;
 *   - num_elements: The name of the struct field that counts the number
 * of struct `field` contiguous elements pointed by struct field `field`.
 * !! The type of the field that counts the number of elements must be a
 * `size_t`. If it is a different size or different type, the behaviour of
 * using the resulting mapper is undefined.
 *   - field_mapper: A pointer to a `struct aml_mapper` that maps
 * this field type.
 **/
#define aml_mapper_decl(name, flags, type, ...)                                \
	CONCATENATE(__AML_MAPPER_DECL_, __AML_MAPPER_DECL_SELECT(__VA_ARGS__)) \
	(name, flags, type, __VA_ARGS__)

/**
 * Declare a static mapper for a structure type that does not need
 * to be descended in the copy. The content of the structure is copied
 * on map.
 * @param[in] name: The result mapper variable name.
 * @param[in] type: The type of structure to map.
 **/
#define aml_final_mapper_decl(name, flags, type)                               \
	struct aml_mapper name =                                               \
	        __AML_MAPPER_INIT(flags, type, 0, NULL, NULL, NULL)

/**
 * Declare a static mapper for a structure type with custom flags.
 * @param[in] name: The result mapper variable name.
 * @param[in] flags: A or combination of flags `AML_MAPPER_FLAG_*`.
 * @param[in] type: The type of the structure to map.
 * @param[in] __VA_ARGS__: Must contain a multiple of 2 or 3 arguments.
 * It must not be empty. See `aml_final_mapper_decl()` for empty `__VA_ARGS__`
 * . If `__VA_ARGS__` contains a multiple of 2 arguments, then it must be a
 * list with the pattern: `field, field_mapper` where:
 *   - field: The name of a field to map in `type` struct.
 *   - field_mapper: A pointer to a `struct aml_mapper` that maps
 * this field type.
 * If `__VA_ARGS__` contains a multiple of 3 arguments, then it must be a list
 * with the pattern: `field, num_elements, field_mapper` where:
 *   - field: The name of a field to map in `type` struct.
 *   - num_elements: The name of the `size_t` field in the structure that
 * gives the number contiguous elements the pointer `field` holds.
 * /!\ The type of the field that counts the number of elements must be a
 * `size_t`. If it is a different size or different type, the behaviour of
 * using the resulting mapper is undefined.
 *   - field_mapper: A pointer to a `struct aml_mapper` that maps
 * this field type.
 **/
#define aml_mapper_decl(name, flags, type, ...)                                \
	CONCATENATE(__AML_MAPPER_DECL_, __AML_MAPPER_DECL_SELECT(__VA_ARGS__)) \
	(name, flags, type, __VA_ARGS__)

/**
 * Declare a static mapper for a structure type that does not need
 * to descend child fields when mapping it in a different memory region.
 * The content of the structure is copied on map.
 * @param[in] name: The result mapper variable name.
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
