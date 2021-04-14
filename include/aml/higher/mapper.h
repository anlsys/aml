/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *******************************************************************************/

#ifndef AML_MAPPER_H
#define AML_MAPPER_H

/**
 * @defgroup aml_mapper "AML Struct Mapper"
 * @brief Hierarchical description of structs.
 *
 * Mapper is construct aimed at holding the description of data structures.
 * A mapper should be able to be instanciated once per `struct` declaration
 * and accurately describe any dynamic instance of the same structure.
 * Mapper can then be used to map/allocate a structure in different memory
 * regions, and copy in between mappings with the associated `dma`.
 * @{
 **/

/**
 * The default mapper behaviour is to only copy pointers indirections
 * to mapped fields of a structure (mapper with child mappers).
 * If this flag is set then the full `src` structure will be copied on map
 * before copying indirection to mapped child fields.
 */
#define AML_MAPPER_FLAG_COPY 0x1

/**
 * The default mapper behaviour is to walk to complete src struct and mappers
 * to compute the total required size and allocate everything in a single packed
 * chunk of memory.
 * If this flag is set then the struct associated with this mapper and children
 * mappers (with flag AML_MAPPER_FLAG_SPLIT unset) is allocated in a separate
 * chunk.
 * There are several use cases for this feature:
 * 1. Breaking down a big allocation in smaller pieces when the allocator
 * does not find one large space to allocate to.
 * 2. Aligning fields with the alignement offered by the allocator.
 */
#define AML_MAPPER_FLAG_SPLIT 0x2

/**
 * The default mapper behaviour is to allocate a new piece of memory for the
 * top level structure of a mapped structure hierarchy.
 * When this flag is set, the `dst` pointer to map is assumed to be already
 * allocated ON HOST, and only its field are to be allocated. This flag can
 * only be used on the top level structure/mapper. If this flag is set in any
 * other mapper of a mapper hierarchy, it is silently ignored.
 */
#define AML_MAPPER_FLAG_SHALLOW 0x4

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
	// An array of function pointers that takes this a pointer to a struct
	// represented by this mapper as input and returns the number of element
	// in each pointer field.
	// Any field with NULL function pointer is considered to have a
	// single element. If the array itself is NULL, then all fields are
	// considered to be single elements.
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
 * same type as it self and mapper pointing at each others, then the behaviour 
 * of this function is undefined. 
 * This function does not implement cycles detection and
 * will loop on cycles, dereferencing data out of memory bounds and eventually
 * loop allocating all memory if a cycle is met.
 * @param[out] out: A pointer to where mapper should be allocated.
 * @param[in] flags: A ORed set of `AML_MAPPER_FLAG_*` to customize mapper
 * behaviour.
 * @see AML_MAPPER_FLAG_COPY
 * @see AML_MAPPER_FLAG_SPLIT
 * @see AML_MAPPER_FLAG_SHALLOW
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

//-----------------------------------------------------------------------------
// Mmap mapped struct.
//-----------------------------------------------------------------------------

/**
 * Perform a deep allocation from host struct to a new struct.
 * The nested structures mapped in this allocation will have their fields
 * pointing to dynamically allocated child structures.
 *
 * There is no implicit synchronization between
 * resulting mapping and source pointer. This is a one time explicit allocation
 * (and possibly copy).
 *
 * This feature requires that the
 * pointer yielded by area can be safely offseted (not dereferenced) from host
 * as long as the result pointer is within the bounds of allocation. If the
 * resulting pointer do not support this property, then using this function is
 * undefined.
 *
 * @param mapper[in]: The mapper describing the struct pointed by `ptr`.
 * @param src[in]: A host pointer on which to perform a deep copy.
 * @param dst[in,out]: If `mapper` has flag `AML_MAPPER_FLAG_SHALLOW` set
 * then dst is a pointer to a memory area on host with at least
 * `mapper->size * num` bytes of space available. If `mapper` has child fields
 * to map, and the mapper to these fields do not have flag
 * `AML_MAPPER_FLAG_SPLIT` set, then `dst` must also secure space for this
 * fields and recursively their child fields under the same condition.
 * Else, when mapper do not set flag `AML_MAPPER_FLAG_SHALLOW`, `dst` is
 * a `(void**)` pointer, and it is set point to the newly mapped structure.
 * @param area[in]: The area where to allocate copy.
 * `area` must yield a pointer on which pointer arithmetic within bounds gives
 * a valid pointer.
 * @param num[in]: The number of contiguous elements represented by `mapper`
 * stored in `src`. For mapping of a single struct, `num` is one. If `src` is an
 * array of `num` structs, then this function will also map an array of `num`
 * structs in `dst`.
 * @param area[in]: The area used to allocate memory in order to store the
 * mapped (array of) structure(s).
 * @see aml_area
 * @param area_opts[in]: Options to provide with area when allocating space.
 * @param dma[in]: A dma engine able to perform movement from host to
 * target `area`.
 * @see aml_dma
 * @param dma_op[in]: A copy operator that performs copy of continuous and
 * contiguous bytes to be used with the `dma` engine.
 * @param dma_op_arg[in]: Optional arguments to pass to the `dma_op` operator on
 * each copy operation.
 * @return AML_SUCCESS on success.
 * @return If any of `area` or `dma` engines return an error, then the function
 * gracefully fails and returns the same error code. In the mean time, `dst`
 * pointer will be left untouched.
 */
int aml_mapper_mmap(struct aml_mapper *mapper,
                    void *src,
                    void *dst,
                    size_t num,
                    struct aml_area *area,
                    struct aml_area_mmap_options *area_opts,
                    struct aml_dma *dma,
                    aml_dma_operator dma_op,
                    void *dma_op_arg);

/**
 * Perform a backward deepcopy from a structure to another.
 * This feature requires that `src` and `dst` pointers can
 * be safely offseted (not dereferenced) from host
 * as long as the result pointer is within the bounds of allocation. If the
 * pointers do not support this property, then using this function is
 * undefined.
 * @param[in] mapper: The description of the structures to copy.
 * @param[in] src: A pointer to a structure accurately described by mapper.
 * @param[in] dst: A pointer to a structure accurately described by mapper.
 * @param num[in]: The number of contiguous elements represented by `mapper`
 * stored in `src`. For copying a single struct, `num` is one. If `src` is an
 * array of `num` structs, then this function will also copy an array of `num`
 * structs in `dst`.
 * @param dma[in]: A dma engine able to perform movement from area of `src` to
 * area of `dst`.
 * @see aml_dma
 * @param dma_op[in]: A copy operator that performs copy of continuous and
 * contiguous bytes to be used with the `dma` engine.
 * @param dma_op_arg[in]: Optional arguments to pass to the `dma_op` operator on
 * each copy operation.
 * @return AML_SUCCESS on success.
 * @return If any of `area` or `dma` engines return an error, then the function
 * fails with eventually some pieces of `src` copie into `dst` and returns the
 * same error code.
 */
int aml_mapper_copy(struct aml_mapper *mapper,
										void *src,
										void *dst,
										size_t num,
										struct aml_dma *dma,
										aml_dma_operator dma_op,
										void *dma_op_arg);

/**
 * Unmap the structure pointed by `ptr`.
 * @param[in] mapper: The description of the mapped structure.
 * @param[in] ptr: The mapped pointer.
 * `ptr` must have been allocated with the same `mapper` and `area`.
 * @param area[in]: The area used to allocate memory in order to store the
 * mapped (array of) structure(s).
 * @see aml_area
 * @param area_opts[in]: Options to provide with area when allocating space.
 * @param dma[in]: A dma engine able to perform movement from device target
 * `area` to host.
 * @see aml_dma
 * @param dma_op[in]: A copy operator that performs copy of continuous and
 * contiguous bytes to be used with the `dma` engine.
 * @param dma_op_arg[in]: Optional arguments to pass to the `dma_op` operator on
 * each copy operation.
 * @return The total size that was unmapped on success.
 * @return An AML error code from `dma` engine on error. If this
 * function fails, the `ptr` to unmap might leak memory.
 */
ssize_t aml_mapper_munmap(struct aml_mapper *mapper,
                          void *ptr,
                          struct aml_area *area,
                          struct aml_dma *dma,
                          aml_dma_operator dma_op,
                          void *dma_op_arg);

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
 * @}
 **/

#endif // AML_MAPPER_H