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
 * regions, and copy in between mappings with the associated `dma`. When
 *buidling a mapper tree of a structure, the process must start from the leaves
 *and go to the root.
 * @{
 **/

// If set then the src structure will be copied on map.
#define AML_MAPPER_FLAG_COPY 0x1

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
 * @param[out] out: A pointer to where mapper should be allocated.
 * @param[in] size: The size of the structure to map.
 * If size is 0, then this struct will not be mapped. However, its
 * fields will if any.
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
 * Perform a deep allocation from host struct to a new empty struct in a an
 * area. If the new empty struct has fields that are dynamically allocated
 * and these fields are described by a list of mappers stored into the parent
 * struct mapper, then the deep allocation will cross the hierarchy of
 * structures mapper to determine the size needed for allocation. The
 * resulting allocation is a single chunk of contiguous memory.
 * The nested structures mapped in this allocation will have their fields
 * pointing to dynamically allocated child structures set to point to a
 * dedicated space in the new allocation. If the mapper of a structure has
 * the flag AML_MAPPER_FLAG_COPY set, then the content of the structure
 * described by the mapper will also be copied but won't overwrite the new
 * indirections of the newly allocated structure.
 *
 * There is no implicit synchronization between
 * resulting mapping and source pointer. This is a one time explicit allocation,
 * and possibly copy.
 *
 * This feature requires that the
 * pointer yielded by area can be safely offseted (not dereferenced) from host
 * as long as the result pointer is within the bounds of allocation. If the
 * resulting pointer do not support this property, then using this function is
 * undefined.
 *
 * If the structure to map is not a tree, e.g it has fields pointing at each
 * others, then the behaviour of this function is undefined.
 *
 * @param mapper[in]: The mapper describing the struct pointed by `ptr`.
 * @param ptr[in]: A host pointer on which to perform a deep copy.
 * @param area[in]: The area where to allocate copy.
 * `area` must yield pointer on which pointer arithmetic within bounds yields
 * a valid pointer.
 * @see aml_area
 * @param opts[in]: Options specific to the area. May be NULL if the area
 * allows it.
 * @param dma[in]: A dma engine able to perform movement from host to
 * target `area`.
 * @see aml_dma
 * @param op[in]: A 1D copy operator for the dma engine.
 * @param op_arg[in]: Optional arguments to pass to the `op` operator.
 * @param size[out]: If not NULL, allocation size is stored here.
 * This may further be used with aml_area_munmap() and `area` to cleanup
 * resulting pointer.
 * @return On success, a pointer to a newly allocated deep copy of source
 * pointer. The result is allocated in a single chunk of memory of the same
 * size as source pointer requires.
 */
void *aml_mapper_mmap(struct aml_mapper *mapper,
                      void *ptr,
                      struct aml_area *area,
                      struct aml_area_mmap_options *opts,
                      struct aml_dma *dma,
                      aml_dma_operator op,
                      void *op_arg,
                      size_t *size);
/**
 * Perform the same work as aml_mapper_mmap() except it does not allocate
 * the top level structure. This is intended for use cases where the top level
 * structure is used on the host and some fields are used on device.
 * @see aml_mapper_mmap()
 * @param mapper[in]: The mapper describing the struct pointed by `ptr`.
 * @param dst[in,out]: The destination pointer to top structure.
 * The pointer must be a host pointer large enough to fit the top level
 * structure. Fields will be allocated and descended if needed.
 * @param src[in]: The host pointer to top level structure to deep copy.
 * @param area[in]: The area where to allocate copy.
 * `area` must yield pointer on which pointer arithmetic within bounds yields
 * a valid pointer.
 * @see aml_area
 * @param opts[in]: Options specific to the area. May be NULL if the area
 * allows it.
 * @param dma[in]: A dma engine able to perform movement from host to
 * target `area`.
 * @see aml_dma
 * @param op[in]: A 1D copy operator for the dma engine.
 * @param op_arg[in]: Optional arguments to pass to the `op` operator.
 * @param size[out]: If not NULL, allocation size is stored here.
 * This may further be used with aml_area_shallow_munmap() and `area` to cleanup
 * resulting pointer.
 */
int aml_mapper_shallow_mmap(struct aml_mapper *mapper,
                            void *src,
                            void *dst,
                            struct aml_area *area,
                            struct aml_area_mmap_options *opts,
                            struct aml_dma *dma,
                            aml_dma_operator op,
                            void *op_arg,
                            size_t *size);

/**
 * Perform a backward deepcopy from a structure to another.
 * This feature requires that `src` and `dst` pointers can
 * be safely offseted (not dereferenced) from host
 * as long as the result pointer is within the bounds of allocation. If the
 * pointers do not support this property, then using this function is
 * undefined.
 * If the structure to copy is not a tree, e.g it has fields pointing at each
 * others, then the behaviour of this function is undefined.
 * @param[in] mapper: The description of the structures to copy.
 * @param[in] src: A pointer to a structure accurately described by mapper.
 * @param[in] dst: A pointer to a structure accurately described by mapper.
 * @param dma[in]: A dma engine able to perform contiguous copy
 * from `src` to `dst`.
 * @see aml_dma
 * @param op[in]: A 1D copy operator for the dma engine.
 * @param op_arg[in] Optional arguments to pass to the `op` operator.
 */
int aml_mapper_copy_back(struct aml_mapper *mapper,
                         void *src,
                         void *dst,
                         struct aml_dma *dma,
                         aml_dma_operator op,
                         void *op_arg);

/**
 * Unmap the structure pointed by ptr in area.
 * It is faster to save pointer size at allocation and use
 * aml_area_munmap() and `area` to cleanup `ptr`.
 * @param[in] mapper: The description of the mapped structure.
 * @param[in] ptr: The mapped pointer.
 * @param[in] area: The area used to map ptr.
 * @param dma[in]: A dma engine able to perform movement from `area` to
 * host.
 * @see aml_dma
 * @param op[in]: A 1D copy operator for the dma engine.
 * @param op_arg[in] Optional arguments to pass to the `op` operator.
 */
void aml_mapper_munmap(struct aml_mapper *mapper,
                       void *ptr,
                       struct aml_area *area,
                       struct aml_dma *dma,
                       aml_dma_operator op,
                       void *op_arg);

/**
 * Unmap the content of structure pointed by ptr in area.
 * The content of ptr must have been previously allocated with
 * aml_mapper_shallow_mmap().
 * @see aml_mapper_shallow_mmap()
 * @param[in] mapper: The description of the mapped structure.
 * @param[in] ptr: The mapped pointer.
 * @param[in] area: The area used to map ptr.
 * @param dma[in]: A dma engine able to perform movement from `area` to
 * host.
 * @param op[in]: A 1D copy operator for the dma engine.
 * @param op_arg[in] Optional arguments to pass to the `op` operator.
 */
void aml_mapper_shallow_munmap(struct aml_mapper *mapper,
                               void *ptr,
                               struct aml_area *area,
                               struct aml_dma *dma,
                               aml_dma_operator op,
                               void *op_arg);
/**
 * Declare a static mapper for a structure type.
 * @param[in] name: The result mapper variable name.
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
 *   - field_mapper: A pointer to a `struct aml_mapper` that maps
 * this field type.
 **/
#define aml_mapper_decl(name, type, ...)                                       \
	CONCATENATE(__AML_MAPPER_DECL_, __AML_MAPPER_DECL_SELECT(__VA_ARGS__)) \
	(name, type, AML_MAPPER_FLAG_COPY, __VA_ARGS__)

#define aml_nocopy_mapper_decl(name, type, ...)                                \
	CONCATENATE(__AML_MAPPER_DECL_, __AML_MAPPER_DECL_SELECT(__VA_ARGS__)) \
	(name, type, 0, __VA_ARGS__)

/**
 * Declare a static mapper for a structure type that does not need
 * to be descended in the copy.
 * @param[in] name: The result mapper variable name.
 * @param[in] type: The type of structure to map.
 **/
#define aml_final_mapper_decl(name, type)                                      \
	struct aml_mapper name = __AML_MAPPER_INIT(AML_MAPPER_FLAG_COPY, type, \
	                                           0, NULL, NULL, NULL)

#define aml_final_nocopy_mapper_decl(name, type)                               \
	struct aml_mapper name = __AML_MAPPER_INIT(0, type, 0, NULL, NULL, NULL)

/**
 * @}
 **/

#endif // AML_MAPPER_H
