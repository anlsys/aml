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
 * Mapper is construct aimed at holding the description of data structures.
 * A mapper should be able to be instanciated once per `struct` declaration
 * and accurately describe any dynamic instance of the same structure.
 * Mapper can then be used to map/allocate a structure in different memory
 * regions, and copy in between mappings with the associated `dma`.
 * @{
 **/

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
 * @param mapper[in]: The mapper describing the struct pointed by `src`.
 * @param dst[in,out]: A pointer (void**) that will be set to the newly
 * allocated and mapped structure.
 * If `mapper` has flag `AML_MAPPER_FLAG_SHALLOW` set
 * then dst is a pointer to a memory area on host with at least
 * `mapper->size * num` bytes of space available. If `mapper` has child fields
 * to map, and the mappers to these fields have flag.
 * @param src[in]: A host pointer on which to perform a deep copy.
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
                    void *dst,
                    void *src,
                    size_t num,
                    struct aml_area *area,
                    struct aml_area_mmap_options *area_opts,
                    struct aml_dma *dma,
                    aml_dma_operator dma_op);

/**
 * Perform a backward deepcopy from a structure to another host structure.
 * @param[in] mapper: The description of the structures to copy.
 * @param[in] dst: A host pointer to a structure accurately described by mapper.
 * @param[in] src: A pointer to a structure accurately described by mapper.
 * `src` pointer must be safely offsetsable from host as long as the result
 * pointer is within the bounds of allocation. If the pointer does not support
 * this property, then using this function is undefined.
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
 * fails with eventually some pieces of `src` copied into `dst` and returns the
 * same error code.
 */
int aml_mapper_copy(struct aml_mapper *mapper,
                    void *dst,
                    void *src,
                    size_t num,
                    struct aml_dma *dma,
                    aml_dma_operator dma_op);

/**
 * Unmap the structure pointed by `ptr`.
 * @param[in] mapper: The description of the mapped structure.
 * @param[in] ptr: The mapped pointer.
 * `ptr` must have been allocated with the same `mapper` and `area`.
 * @param[in] num: The number of contiguous elements stored in `ptr`, described
 * by `mapper`. This value is usually 1.
 * @param[in] src: The original pointer used to map `ptr`. This is required
 * in order to compute array fields length without performing extra DMAs and
 * rebuilding locally the equivalent of `src`.
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
                          size_t num,
                          void *src,
                          struct aml_area *area,
                          struct aml_dma *dma,
                          aml_dma_operator dma_op);

//-----------------------------------------------------------------------------
// Default Mappers
//-----------------------------------------------------------------------------

/** Default mapper for elements of type char */
aml_final_mapper_decl(aml_char_mapper, 0, char);
/** Default mapper for elements of type short */
aml_final_mapper_decl(aml_short_mapper, 0, short);
/** Default mapper for elements of type int */
aml_final_mapper_decl(aml_int_mapper, 0, int);
/** Default mapper for elements of type long */
aml_final_mapper_decl(aml_long_mapper, 0, long);
/** Default mapper for elements of type long long */
aml_final_mapper_decl(aml_long_long_mapper, 0, long long);
/** Default mapper for elements of type unsigned char */
aml_final_mapper_decl(aml_uchar_mapper, 0, unsigned char);
/** Default mapper for elements of type unsigned int */
aml_final_mapper_decl(aml_uint_mapper, 0, unsigned int);
/** Default mapper for elements of type unsigned long */
aml_final_mapper_decl(aml_ulong_mapper, 0, unsigned long);
/** Default mapper for elements of type unsigned long long */
aml_final_mapper_decl(aml_ulong_long_mapper, 0, unsigned long long);
/** Default mapper for elements of type float */
aml_final_mapper_decl(aml_float_mapper, 0, float);
/** Default mapper for elements of type double */
aml_final_mapper_decl(aml_double_mapper, 0, double);
/** Default mapper for elements of type long double */
aml_final_mapper_decl(aml_long_double_mapper, 0, long double);
/** Default mapper for pointer elements */
aml_final_mapper_decl(aml_ptr_mapper, 0, void *);

/** Default mapper in a seperate allocation for elements of type char */
aml_final_mapper_decl(aml_char_split_mapper, AML_MAPPER_FLAG_SPLIT, char);
/** Default mapper in a seperate allocation for elements of type short */
aml_final_mapper_decl(aml_short_split_mapper, AML_MAPPER_FLAG_SPLIT, short);
/** Default mapper in a seperate allocation for elements of type int */
aml_final_mapper_decl(aml_int_split_mapper, AML_MAPPER_FLAG_SPLIT, int);
/** Default mapper in a seperate allocation for elements of type long */
aml_final_mapper_decl(aml_long_split_mapper, AML_MAPPER_FLAG_SPLIT, long);
/** Default mapper in a seperate allocation for elements of type long long */
aml_final_mapper_decl(aml_long_long_split_mapper,
                      AML_MAPPER_FLAG_SPLIT,
                      long long);
/** Default mapper in a seperate allocation for elements of type unsigned char
 */
aml_final_mapper_decl(aml_uchar_split_mapper,
                      AML_MAPPER_FLAG_SPLIT,
                      unsigned char);
/** Default mapper in a seperate allocation for elements of type unsigned short
 */
aml_final_mapper_decl(aml_ushort_split_mapper,
                      AML_MAPPER_FLAG_SPLIT,
                      unsigned short);
/** Default mapper in a seperate allocation for elements of type unsigned int */
aml_final_mapper_decl(aml_uint_split_mapper,
                      AML_MAPPER_FLAG_SPLIT,
                      unsigned int);
/** Default mapper in a seperate allocation for elements of type unsigned long
 */
aml_final_mapper_decl(aml_ulong_split_mapper,
                      AML_MAPPER_FLAG_SPLIT,
                      unsigned long);
/** Default mapper in a seperate allocation for elements of type unsigned long
 * long */
aml_final_mapper_decl(aml_ulong_long_split_mapper,
                      AML_MAPPER_FLAG_SPLIT,
                      unsigned long long);
/** Default mapper in a seperate allocation for elements of type float */
aml_final_mapper_decl(aml_float_split_mapper, AML_MAPPER_FLAG_SPLIT, float);
/** Default mapper in a seperate allocation for elements of type double */
aml_final_mapper_decl(aml_double_split_mapper, AML_MAPPER_FLAG_SPLIT, double);
/** Default mapper in a seperate allocation for elements of type long double */
aml_final_mapper_decl(aml_long_double_split_mapper,
                      AML_MAPPER_FLAG_SPLIT,
                      long double);
/** Default mapper in a seperate allocation for pointer elements */
aml_final_mapper_decl(aml_ptr_split_mapper, AML_MAPPER_FLAG_SPLIT, void *);

//------------------------------------------------------------------------------
// Mapper iterator.
//------------------------------------------------------------------------------

struct aml_mapper_iterator {
	// Base pointer iterated with this iterator.
	void *base_ptr;
	// Mapper of this base pointer
	struct aml_mapper *mapper;
	// The memory size iterated so far.
	size_t tot_size;
	// The current mapper field iterated.
	size_t field_num;
	// If base_ptr is an array, this is the iteration index in array.
	size_t array_num;
	// If base_ptr is an array, this is the number of array elements.
	size_t array_size;
	// dma engine from base_ptr area to host to dereference pointers.
	struct aml_dma *dma;
	// Memcpy operator of dma engine.
	aml_dma_operator memcpy_op;
	// Position of this iterator pointer in the stack.
	size_t stack_pos;
	// Number of iterator pointers in the stack.
	size_t stack_size;
};

/**
 * Create an iterator over the struct pointed by `ptr` and described
 * by `mapper`. On each iteration, this iterator will return pointer
 * to a child field in the structure hierarchy. If a child field is
 * an array of structs, then iteration will also iterate over elements
 * of the array. Iterations will also update `tot_size` field of iterator
 * computing the size of the iterated struct. At the last valid iteration
 * step, `tot_size` contains the complete structure size.
 *
 * @see mapper_iter_next()
 *
 * @param[out] out: Where to store pointer to allocated iterator.
 * @param[in] ptr: The pointer to the structure to iterate.
 * @param[in] mapper: The mapper describing the structure.
 * @param[in, out] dma: The dma engine from `ptr` area to host used to
 * dereference `ptr` and child fields. if `dma` is NULL, `ptr` is assumed
 * to be a host pointer.
 * @param[in] memcpy_op: A memcpy operator taking raw pointers and their
 * size as input.
 *
 * @return -AML_EINVAL if out, ptr or mapper are NULL.
 * @return -AML_ENOMEM if there was not enough memory to allocate iterator.
 * @return AML_SUCCESS on success.
 */
int aml_mapper_iterator_create(struct aml_mapper_iterator **out,
                               void *ptr,
                               struct aml_mapper *mapper,
                               struct aml_dma *dma,
                               aml_dma_operator memcpy_op);

/**
 * Store struct pointer reflected by current iterator position and
 * forward iterator.
 * Iterator forwarding works as follow:
 * 1. If the current structure field can be forwarded, then it is forwarded
 * and next field is descended by dereferencing it and updating iterator
 * pointer to the iterator representing iteration at current structure
 * depth.
 * @see mapper_iter_next_field()
 * 2. Else, if current iterator represents an array of structs, and
 * the structs have children, then, the iterator is forwarded to next array
 * element and field cursor is reset to point to first struct field of
 * current element.
 * @see mapper_iter_next_element()
 * 3. If none of 1. and 2. can be forwarded, then this structure has been
 * completely visited and we move on to parent. We resume step 1. from
 * parent. If the result of step 3. is pointing to the root structure, then
 * the entire structure has been visited and the iterator is freed and
 * stops.
 * @see mapper_iter_parent_struct()
 *
 * @param[in, out] it: The current iteration step. This pointer might be
 * reallocated and updated if the next step is to descend a child field.
 * @param[out] out: Where to store the resulting struct pointer of next
 * iteration.
 *
 * @return AML_EDOM with `*out` set to `NULL` if last iteration step has
 * been reached.
 * @return AML_SUCCESS with `*out` pointing to the next substructure on
 * success.
 * @return -AML_EINVAL if `it` or `out` are `NULL`.
 * @return -AML_ENOMEM if iteration required to allocate more memory
 * and allocation failed.
 * @return A dma error code if a dma operation was required and failed.
 **/
int aml_mapper_iter_next(struct aml_mapper_iterator **it, void **out);

/**
 * Forward iterator to next field if possible.
 * If current struct pointed by iterator has fields and the iterator
 * field cursor is not pointing after last field, then the pointed field
 * is descended. The corresponding pointer is dereferenced to obtain
 * pointer to child struct using embedded dma engine if needed.
 * The resulting pointer is stored in `out`.
 * The input iterator field cursor is forwarded in the process.
 * When descending iterator is updated to an iterator respresenting
 * child structure in the stack of parent structures iterators.
 * The stack of iterators might be reallocated and if reallocation fails,
 * nothing is done and -AML_ENOMEM is returned.
 * If current structure has no child or iterator field cursor is pointing
 * after last child, then nothing is done and AML_EDOM is returned.
 *
 * @param [in,out] it: A pointer to iterator representing the state
 * of current iteration. This pointer might be updated and even reallocated
 * when this function is called.
 * @param[out] out: Where to store pointer to next child field.
 *
 * @return AML_EDOM if last field iteration of current structure has been
 * reached.
 * @return AML_SUCCESS with `*out` pointing to the next child field on
 * success.
 * @return -AML_EINVAL if `it` or `out` are `NULL`.
 * @return -AML_ENOMEM if iteration required to allocate more memory
 * and allocation failed.
 * @return A dma error code if a dma operation was required and failed.
 */
int aml_mapper_iter_next_field(struct aml_mapper_iterator **it, void **out);

/**
 * Forward iterator over an array of structs.
 * If iterator array cursor points at last array element, then nothing is
 * done and `AML_EDOM` is returned.
 * If structs of the array of struct pointed by iterator are final structs
 * (they have no child fields) then nothing is done and `AML_EDOM` is
 * returned.
 * Else, iterator array cursor is forwarded, iterator field cursor is reset
 * and pointer to the next array element is returned in `out`.
 *
 * @param [in,out] it: A pointer to iterator representing the state
 * of current iteration.
 * @param[out] out: Where to store pointer to next array element.
 *
 * @return AML_EDOM if last array element iteration of current structure
 * has been reached.
 * @return AML_SUCCESS with `*out` pointing to the next array element on
 * success.
 * @return -AML_EINVAL if `it` or `out` are `NULL`.
 */
int aml_mapper_iter_next_element(struct aml_mapper_iterator **it, void **out);

/**
 * Stop iterating current structure and move iterator cursosr to parent
 * structure.
 *
 * @param [in,out] it: A pointer to iterator representing the state
 * of current iteration. `it` is updated after this function call.
 * @param[out] out: Where to store pointer to parent struct.
 *
 * @return AML_EDOM if current iteration struct is the root struct.
 * @return AML_SUCCESS with `*out` pointing to the parent struct on
 * success.
 * @return -AML_EINVAL if `it` or `out` are `NULL`.
 */
int aml_mapper_iter_parent_struct(struct aml_mapper_iterator **it, void **out);

/**
 * @}
 **/

#ifdef __cplusplus
}
#endif
#endif // AML_MAPPER_H
