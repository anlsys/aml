/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#ifndef AML_MAPPER_CREATOR_H
#define AML_MAPPER_CREATOR_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup aml_mapper_creator "AML Mapper Creator"
 * @brief Base facillity to copy hierarchical structures.
 *
 * @{
 **/

/**
 * The creator structure is a special structure visitor that copies the
 * structure it visits as it is visiting it.
 * It is meant to be used to implement the copy of complex
 * structures to a complex topology of memories.
 *
 * The copy process can be summarized as copying the source structure
 * bit by bit on the host, overwriting fields pointers on host to point
 * to the device memory where the corresponding field will be copied, and
 * finally copying the big host buffer to the device. The resulting copy,
 * is packed in a contiguous buffer.
 *
 * In more details, the creator structure will perform a depth first visit
 * of the source tree-like structure in successive steps.
 * For each step, the creator state matches either a field in a parent
 * structure or an element of an array field if the array field is an
 * array of structs with descendants.
 *
 * When a step is performed, the current structure referenced by the creator
 * in its current state is copied on the host in a buffer large enough to
 * contain the whole structure to copy.
 * The field of the host copy of the parent structure is overwritten with
 * the pointer of the device memory where the current field will be copied
 * in one big copy at the very end.
 *
 * Since the creator state refers to the structure that is about to be
 * copied, the user may perform one of these two actions:
 * - Copy the field on host in the current buffer containing all previously
 * visited fields and overwrite parent pointer to this field to point to
 * device memory where this field will be copied, and then move on to the
 * next field;
 * - "Branch out" by creating a new creator at the current point of visit
 * that will map the current field and its descendants in a different
 * host buffer and device mapped memory.
 *
 * Host buffer and device buffer are allocated to fit just enough data to
 * map the structure, which is the size computed by a visitor that does not
 * account for fields with a mapper that indicates to split the allocation.
 * Therefore, when encountering such a flag in the constructor, the user
 * can only take the "branch-out" option.
 *
 * @see `struct aml_mapper`
 * @see `struct aml_mapper_visitor_state`
 */
struct aml_mapper_creator {
	// Information on current mapper, parent fields and relationship of
	// ancestors with their parent.
	// Note that the field device_ptr of stack, does not refer to a pointer
	// in `device_memory` below but instead to a pointer of the structure
	// that is being copied.
	struct aml_mapper_visitor_state *stack;
	// Target memory where host_memory is being copied at "finish()".
	void *device_memory;
	// Host buffer storing the ongoing construction of the structure
	// to map. At the end of the construction, the buffer is copied to
	// the device memory.
	void *host_memory;
	// Offset in `host_memory` and `device_memory` buffers pointing to the
	// beginning of free space.
	size_t offset;
	// Dma engine to copy from the source structure pointer to host.
	// If the source struture is on host, this pointer can be NULL.
	struct aml_dma *dma_src_host;
	// Dma engine to copy from host to destination structure pointer.
	struct aml_dma *dma_host_dst;
	// Dma memcpy operator to copy from the source structure pointer to
	// host. If the source struture in on host, this pointer can be NULL.
	aml_dma_operator memcpy_src_host;
	// Dma memcpy operator to copy from host to destination structure
	// pointer.
	aml_dma_operator memcpy_host_dst;
};

/**
 * Allocate and instanciate a mapper creator to deep-copy a
 * tree-like structure.
 *
 * Allocation will allocate two buffers, one in host memory and one in
 * target `area`. Buffers will contain a packed copy of the entire
 * structure.
 *
 * `dma_src_host` will be used to copy from the source structure
 * `src_ptr` to the host buffer, while `dma_host_dst` will be use to copy
 * the host buffer in the destination area buffer.
 *
 * After this call succeed, the created mapper creator, will be in a ready
 * state, ready to be iterated with `aml_mapper_creator_next()` to copy the
 * source structure bit by bit on host, and copied to the destination area
 * at the end of the iteration process with `aml_mapper_creator_finish()`.
 *
 * @param[out] out: A pointer where to store the instanciated mapper
 * creator.
 * @param[in] src_ptr: A pointer to the structure to copy. This structure
 * will not be modified in the copy process.
 * @param[in] mapper: The description of the structure pointed by `src_ptr`.
 * @param[in] area: The area used to allocate the space where the source
 * structure will be copied.
 * @param[in] area_opts: Options to configure area behavior.
 * @param[in] dma_src_host: A copy engine to copy from the source pointer to
 * the host executing this call.
 * @param[in] dma_host_dst: A copy engine to copy from the host executing
 * this call to a destination area buffer.
 * @param[in] memcpy_src_host: The memcpy operator of `dma_src_host`.
 * @param[in] memcpy_host_dst: The memcpy operator of `dma_host_dst`.
 * @return AML_SUCCESS on success.
 * @return -AML_ENOMEM if there was not enough memory to allocate host
 * buffer or to perform necessary operations.
 * @return `aml_errno` if `aml_area_mmap()` call fails.
 * @return Any error coming `dma_src_host` copied to obtain the source
 * structure total size.
 */
int aml_mapper_creator_create(struct aml_mapper_creator **out,
                              void *src_ptr,
															size_t size,
                              struct aml_mapper *mapper,
                              struct aml_area *area,
                              struct aml_area_mmap_options *area_opts,
                              struct aml_dma *dma_src_host,
                              struct aml_dma *dma_host_dst,
                              aml_dma_operator memcpy_src_host,
                              aml_dma_operator memcpy_host_dst);

/**
 * This function is called to conclude a copy performed with a mapper
 * creator. On success, this function will also take care of cleaning up the
 * resource creator.
 *
 * It will perform a copy of the host copy with embedded destination areas
 * pointers to the destination area.
 * After the copy succeed, the resources associated with the mapper creator
 * are freed and the structure copy in the destination area is returned.
 * The structure will be packed in a single buffer. The size of the buffer
 * is also returned and can be used with the area used to create the
 * destination structure to free the latter.
 *
 * This function should be called only after a call to
 * `aml_mapper_creator_next()` with the same mapper creator returned
 * `-AML_EDOM` meaning that everything has been copied from source
 * structure to the host.
 *
 * @param[in] c: A mapper creator that finished copying and packing source
 * structure on host. This happens when `aml_mapper_creator_next()` with the
 * same mapper creator returns `-AML_EDOM`.
 * @param[out] ptr: Where to store the pointer to the copy of the copied
 * structure.
 * @param[out] size: Where to store the size of the copied structure.
 * @return AML_SUCCESS on success.
 * @return Any error from the dma engine used to copy from host to
 * destination area.
 */
int aml_mapper_creator_finish(struct aml_mapper_creator *c,
                              void **ptr,
                              size_t *size);

/**
 * Perform the next step of the copy from source structure to host buffer.
 *
 * The function copies the current visited field to host, then move to the
 * next field to copy.
 * The next field to copy is obtained doing a depth-first visit as follow:
 * 1. Go to the first field.
 * 2. If there is no first field, go to next array element.
 * 3. If there is no next array element, go to next sibling field.
 * 4. If there is no sibling field, go to parent and then go to 2.
 *
 * If the current state of the creator holds a flag `AML_MAPPER_FLAG_SPLIT`,
 * then the function will do nothing and return `-AML_EINVAL`. This flags
 * means that the structure allocation must be broke at this point and there
 * won't be enough room in the creator buffers to fit current field.
 * In this case, the user should use `aml_mapper_creator_branch()`.
 *
 * @param[in, out] c: A mapper creator representing the current state of a
 * structure copy from a source pointer to the host.
 * @return AML_SUCCESS on success to process this step.
 * @return -AML_EINVAL if the current field has the flag
 * `AML_MAPPER_FLAG_SPLIT` set. In that case, the next call should be
 * `aml_mapper_creator_branch()`.
 * @return -AML_EDOM there is no next field to copy. In that case,
 * the next call should be `aml_mapper_creator_finish()`.
 * @return Any error from dma_src_host arising from copying source pointer
 * to host.
 *
 * @see `aml_mapper_creator_branch()`
 * @see `aml_mapper_creator_finish()`
 */
int aml_mapper_creator_next(struct aml_mapper_creator *c);

/**
 * Create a new mapper creator starting from the current creator state and
 * connect the corresponding field and its descendants to the parent
 * structure in the current mapper creator.
 *
 * Once the new creator is successfully initialized, it can be iterated
 * independently and in a thread safe manner from the initial creator.
 *
 * This function should be called after `aml_mapper_creator_next()` returned
 * `-AML_EINVAL` on the current mapper creator. After this function succeed,
 * the user can resume calling `aml_mapper_creator_next()` on the initial
 * creator to continue mapping parent structure. The whole structure, will
 * be entirely mapped only when both the current creator and the new
 * creator are finished.
 *
 * @param[out] out: A pointer where to store the new mapper creator.
 * @param[in] c: The current mapper creator which should be in a state with
 * the flag `AML_MAPPER_FLAG_SPLIT` set.
 * @param[in] area: The area where to copy the structure of the new creator.
 * @param[in] area_opts: Options for customizing area behavior.
 * @param[in] dma_host_dst: A dma engine to copy data from host to `area`.
 * @param[in] memcpy_host_dst: The memcpy operator for the dma engine.
 *
 * @return AML_SUCCESS on success.
 * @return -AML_EDOM if there is nothing left to copy in the current
 * creator after this call. In that case, the next call on current creator
 * should be `aml_mapper_creator_finish()`.
 * @return Any error from `aml_mapper_creator_create()` if the creation of
 * the new creator failed.
 * @return -AML_ENOMEM, even if the creation of the new creator succeeded,
 * when moving current creator to the next field afterward, if a child needs
 * to be descended but there is not enough memory to allocate the
 * corresponding state. This is very unlikely to happen because descending
 * a child field also means that the current creator had to go up just
 * before and therefore, freed memory for a state.
 *
 * @see `aml_mapper_creator_create()`
 */
int aml_mapper_creator_branch(struct aml_mapper_creator **out,
                              struct aml_mapper_creator *c,
                              struct aml_area *area,
                              struct aml_area_mmap_options *area_opts,
                              struct aml_dma *dma_host_dst,
                              aml_dma_operator memcpy_host_dst);

/**
 * @}
 **/

#ifdef __cplusplus
}
#endif
#endif // AML_MAPPER_CREATOR_H
