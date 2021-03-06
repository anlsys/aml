/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#ifndef AML_MAPPER_VISITOR_H
#define AML_MAPPER_VISITOR_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @addtogroup aml_mapper
 *
 * AML mapper visitor defines the structures and method to walk a a structure
 * described by a mapper.
 *
 * @{
 **/

/**
 * The current state of a mapper visitor for the depth being visited.
 * @see `aml_mapper_visitor`
 */
struct aml_mapper_visitor_state {
	// Device pointer of the element currrently visited.
	void *device_ptr;
	// Byte copy of the element.
	// This field is used to read fields pointer and the number of
	// elements if the structure is actually an array of structures.
	// The copy from device_ptr only happens when a parent descends its
	// first child.
	void *host_copy;
	// Mapper of the element currrently visited.
	struct aml_mapper *mapper;
	// The field index in parent structure described by parent mapper.
	// This field matches the field number in parent mapper.
	size_t field_num;
	// If the field currently visited is part of an array, this
	// is the number of elements in this array. If not, it is 0.
	size_t array_size;
	// If the field currently visited is part of an array, this
	// is the index of the current element in the array.
	// The pointer of the array beginning is therefore:
	// `device_ptr - mapper->size * array_num`.
	size_t array_num;
	// Parent states from parent data structures.
	struct aml_mapper_visitor_state *next;
};

/**
 * The visitor structure maintains the state of a visit of a data structure
 * described by a hierarchy of mappers. The visited data structure, may have
 * references to itself (cycles). However, cycles are not detected and might
 * be leading to an infinite walk. The visited data structure is not modified
 * by the visit and must not undergo structural modifications that affect the
 * visit during the latter.
 *
 * For a given step/state of the visit, it is possible to query the pointer of
 * the data being visited from inside the visited data structure with
 * `aml_mapper_visitor_ptr()`. The latter can be the root structure
 * at the creation of the visitor, a field inside the data structure when
 * visiting fields, or an array of elements when visiting elements of a field
 * which is an array of contiguous elements.
 *
 * It is also possible to query the size in bytes of the visited field with
 * `aml_mapper_visitor_size()`. Unlike the mapper size field that only accounts
 * for the top level structure, this size accounts for the top level structure
 * size and all its descendants.
 *
 * Visiting can be done toward a parent structure
 * (`aml_mapper_visitor_parent()`), the first child field of the structure
 * (`aml_mapper_visitor_first_field()`), sibling fields of the same parent
 * (`aml_mapper_visitor_next_field()`) or even across array elements of an
 * array field (`aml_mapper_visitor_next_array_element()`).
 *
 * @see `aml_mapper`
 */
struct aml_mapper_visitor {
	// Stack of visited fields where head is the currently visited field
	// and next elements are successive parents of the currently visited
	// field.
	struct aml_mapper_visitor_state *stack;
	// dma engine from base_ptr area to host to dereference pointers.
	struct aml_dma *dma;
	// Memcpy operator of dma engine.
	aml_dma_operator memcpy_op;
};

/**
 * Create a visitor of a structure pointed by `ptr` and described by a `mapper`.
 *
 * If the structure pointer cannot be read from the host, it must be copied
 * piece by piece during the visit, from the memory where it lives to the host
 * memory with a `dma` engine and a `memcpy` operator. However, if the pointer
 * is readable from the host, it is better to set these two arguments to
 * NULL to avoid unnecessary copies.
 *
 * @param[out] out: A pointer where to store the newly allocated visitor.
 * @param[in] ptr: The pointer to the structure to visit.
 * @param[in] mapper: The description of the structure to visit.
 * @param[in] dma: A dma engine to copy data from where ptr lives to the host.
 * @param[in] memcpy_op: The suitable `memcpy` operator for the dma engine.
 * @return AML_SUCCESS on sucess.
 * @return -AML_ENOMEM on failure due to host memory allocation failure.
 */
int aml_mapper_visitor_create(struct aml_mapper_visitor **out,
                              void *ptr,
                              struct aml_mapper *mapper,
                              struct aml_dma *dma,
                              aml_dma_operator memcpy_op);

/**
 * Release memory allocated to a visitor.
 * The visitor `it` cannot be used after this call.
 *
 * @param[in, out] it: The visitor to free.
 * @return AML_SUCCESS on success.
 * @return -AML_EINVAL if it is NULL.
 */
int aml_mapper_visitor_destroy(struct aml_mapper_visitor *it);

/**
 * Make a visitor from current visited state and limit it to visit only
 * descendants of the current field.
 *
 * @param[in] it: The visitor to copy.
 * @param[out] subtree_visitor: A pointer where to store the new visitor.
 * @return AML_SUCCESS on success.
 * @return -AML_ENOMEM if the new visitor could not be allocated.
 */
int aml_mapper_visitor_subtree(const struct aml_mapper_visitor *it,
                               struct aml_mapper_visitor **subtree_visitor);

/**
 * Go to next array element of this field.
 * After this function succeeds, the pointer returned by
 * `aml_mapper_visitor_ptr()` will point to the next array element of the
 * previously pointed element inside the array.
 *
 * @param[in, out] it: The visitor of the structure being visited.
 * @return AML_SUCCESS on success.
 * @return -AML_EDOM if the field is not an array or if their is no next
 * element.
 * @return -AML_EINVAL if the current element points to NULL.
 */
int aml_mapper_visitor_next_array_element(struct aml_mapper_visitor *it);

/**
 * Go to previous array element of this field.
 * After this function succeeds, the pointer returned by
 * `aml_mapper_visitor_ptr()` will point to the previous array element of the
 * previously pointed element inside the array.
 *
 * @param[in, out] it: The visitor of the structure being visited.
 * @return AML_SUCCESS on success.
 * @return -AML_EDOM if the field is not an array or if their is no previous
 * element.
 * @return -AML_EINVAL if the current element points to NULL.
 */
int aml_mapper_visitor_prev_array_element(struct aml_mapper_visitor *it);

/**
 * Go to next sibling field of the current field from the parent structure.
 *
 * @param[in, out] it: The visitor of the structure being visited.
 * @return AML_SUCCESS on success.
 * @return -AML_EDOM if their is no next sibling field.
 */
int aml_mapper_visitor_next_field(struct aml_mapper_visitor *it);

/**
 * Go to previous sibling field of the current field from the parent structure.
 *
 * @param[in, out] it: The visitor of the structure being visited.
 * @return AML_SUCCESS on success.
 * @return -AML_EDOM if their is no previous sibling field.
 */
int aml_mapper_visitor_prev_field(struct aml_mapper_visitor *it);

/**
 * Go to first field of the current structure.
 *
 * This function might be expensive if the data being visited requires a dma
 * engine to be read. Indeed, the new visited element will need to read
 * the parent element to obtain its own pointer value and compute the number
 * of elements it holds if it is an array. Reading the parent structure may mean
 * transferring data from device to host if the structure being visited does not
 * live on the host.
 *
 * @param[in, out] it: The visitor of the structure being visited.
 * @return AML_SUCCESS on success.
 * @return -AML_EDOM if the current structure has no field to be descended.
 * @return -AML_ENOMEM if the state of the new child field cannot be allocated.
 * @return -AML_EINVAL if the currently visited element points to NULL.
 */
int aml_mapper_visitor_first_field(struct aml_mapper_visitor *it);

/**
 * Go to parent structure of the current structure.
 *
 * @param[in, out] it: The visitor of the structure being visited.
 * @return AML_SUCCESS on success.
 * @return -AML_EDOM if the current structure has no parent that has already
 * been visited.
 */
int aml_mapper_visitor_parent(struct aml_mapper_visitor *it);

/**
 * Get a pointer to the currently visited element from the visited data
 * structure.
 *
 * @param[in] it: The visitor of the structure being visited.
 */
void *aml_mapper_visitor_ptr(struct aml_mapper_visitor *it);

/**
 * Return whether the currently visited element is an array.
 * This is equivalent to `it->stack->array_size > 1`.
 *
 * @param[in] it: The visitor of the structure being visited.
 */
int aml_mapper_visitor_is_array(struct aml_mapper_visitor *it);

/**
 * Return the length of the currently visited array element.
 * If the element is not an array, the len will be 1.
 * This is equivalent to `it->stack->array_size`.
 *
 * @param[in] it: The visitor of the structure being visited.
 */
size_t aml_mapper_visitor_array_len(struct aml_mapper_visitor *it);

/**
 * Compute the size of a structure based on a visitor.
 * Visitor will skip nodes with mapper flag `AML_MAPPER_FLAG_SPLIT` set.
 *
 * @param[in] visitor: A visitor from where to compute size.
 * @param[out] size: A pointer where to store size.
 * @return AML_SUCCESS on success with size set.
 * @return -AML_ENOMEM if allocation during the visit failed.
 * @return -AML_EINVAL if NULL pointer node was found during the visit.
 * @return Any error from the `visitor` dma engine.
 */
int aml_mapper_visitor_size(const struct aml_mapper_visitor *visitor,
                            size_t *size);

/**
 * Check that the structure visited by two different visitors or equal
 * byte-wise except for the value of pointers to child fields.
 *
 * @param[in] lhs: The visitor of the left hand side structure being
 * compared.
 * @param[in] rhs: The visitor of the right hand side structure being
 * compared.
 * @return 1 If structures match.
 * @return 0 If structures don't match.
 * @return -AML_ENOMEM if allocation during the visit failed.
 * @return -AML_EINVAL if NULL pointer node was found during the visit.
 * @return Any error from the visitors dma engine.
 */
int aml_mapper_visitor_match(const struct aml_mapper_visitor *lhs,
                             const struct aml_mapper_visitor *rhs);
/**
 * @}
 **/

#ifdef __cplusplus
}
#endif
#endif // AML_MAPPER_VISITOR_H
