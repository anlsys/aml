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
 * @addtogroup aml_mapper
 *
 * @{
 **/

/**
 * The creator structure is a special structure visitor that copies the
 * structure it visits as it is visiting it.
 * It is meant to be used to implement the copy of complex
 * structures with members in different memories.
 *
 * Note that the source structure
 * must have a tree shape. At the moment, the creator does not support
 * self references and multiple pointers to the same element.
 *
 * The copy process can be summarized as copying the source structure
 * bit by bit on the host, overwriting fields pointers on host to point
 * to the device memory where the corresponding field will be copied, and
 * finally copying the host buffer to the device. The resulting copy,
 * is packed in a contiguous buffer.
 *
 * In more details, the creator structure will perform a depth first visit
 * of the source structure in successive steps.
 * For each step, the creator state matches either a field in a parent
 * structure or an element of an array field if the array field is an
 * array of structs with descendants.
 *
 * When a step is performed, the current structure referenced by the creator
 * in its current state is copied on the host in a buffer large enough to
 * contain the whole structure to copy.
 * The field of the host copy of the parent structure is overwritten with
 * the pointer of the device memory where the current field will be copied
 * in a single copy to the destination pointer at the very end.
 *
 * Since the creator state refers to the structure that is about to be
 * copied, the user may perform one of these three actions:
 * - Copy the field on host and overwrite parent pointer to this field
 * (on the host) to point to the device memory where this field will be copied,
 * and then move on to the next field;
 * - "Branch out" by creating a new creator at the current point of visit
 * that will map the current field and its descendants in a different
 * host buffer and device mapped memory.
 * - Connect a child field that is already to its parent instead of
 * "branching out". The child field is assumed to be a valid instanciation of
 * a structure described the mapper of the creator in its current state.
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
	// Information on the current mapper, parent fields and relationship of
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
	// device_memory and host_memory size in bytes.
	size_t size;
	// Offset in `host_memory` and `device_memory` buffers pointing to the
	// beginning of free space.
	size_t offset;
	// Area where device_memory is allocated.
	// NULL if device_memory is the same as hsot memory.
	struct aml_area *device_area;
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
 * `src_ptr` to the host buffer, while `dma_host_dst` will be used to copy
 * the host buffer to the destination area buffer.
 *
 * After this call succeeds, the created mapper creator, will be in a ready
 * state, ready to be iterated with `aml_mapper_creator_next()` to copy the
 * source structure bit by bit on host. The final structure, will be copied to
 * the destination `area` buffer, after iteration is finished when calling
 * `aml_mapper_creator_finish()`.
 *
 * @param[out] out: A pointer where to store the instanciated mapper
 * creator.
 * @param[in] src_ptr: A pointer to the structure to copy. This structure
 * will not be modified in the copy process.
 * @param[in] size: A size large enough to fit the first chunk of the
 * structure to copy. If no mapper in the hierarchy has the flag
 * `AML_MAPPER_FLAG_SPLIT` set, this is the total size of the structure.
 * `size` can be set to 0 if this not known at the time of the call.
 * If `size` is 0, then the source structure will be visited to obtain this
 * size.
 * @param[in] mapper: The description of the structure pointed by `src_ptr`.
 * @param[in] area: The area used to allocate the space where the source
 * structure will be copied. `area` can be NULL if mapper flag
 * `AML_MAPPER_FLAG_HOST` is set, meaning that no allocation but the host copy
 * is performed.
 * @param[in] area_opts: Options to configure area behavior.
 * @param[in] dma_src_host: A copy engine to copy from the source pointer to
 * the host executing this call. `dma_src_host` can be NULL if the source
 * pointer is already on host.
 * @param[in] dma_host_dst: A copy engine to copy from the host executing
 * this call to a destination area buffer. `dma_host_dst` can be `NULL` if
 * mapper flag `AML_MAPPER_FLAG_HOST` is set, meaning that the target copy is
 * the internal host copy.
 * @param[in] memcpy_src_host: The memcpy operator of `dma_src_host`.
 * It can be NULL if `dma_src_host` is `NULL`.
 * @param[in] memcpy_host_dst: The memcpy operator of `dma_host_dst`.
 * It can be NULL if `dma_host_dst` is `NULL`.
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
 * resource used by the creator, except of course the device pointer holding
 * the copy.
 *
 * It will perform a copy of the host copy with embedded destination areas
 * pointers to the destination area.
 * After the copy succeed, the resources associated with the mapper creator
 * are freed and the structure copy in the destination area is returned.
 * The structure will be packed in a single buffer. The size of the buffer
 * is also returned and can be used with the area used to create the
 * destination structure to free the latter.
 *
 * This function may only be called after a call to
 * `aml_mapper_creator_next()`, `aml_mapper_creator_connect()` or
 * `aml_mapper_creator_branch()` with the same mapper creator returned
 * `-AML_EDOM` meaning that everything has been copied from source
 * structure to the host and the creator is ready to finish the job.
 *
 * @param[in] c: A mapper creator that finished copying and packing source
 * structure on host.
 * @param[out] ptr: Where to store the pointer to the copy of the copied
 * structure.
 * @param[out] size: Where to store the size of the copied structure.
 * @return AML_SUCCESS on success.
 * @return -AML_EINVAL if the user is attempting to finish a creator without
 * copying everything there is to copy. If this is intended, use
 * `aml_mapper_creator_abort()` instead.
 * @return Any error from the dma engine used to copy from host to
 * destination area.
 */
int aml_mapper_creator_finish(struct aml_mapper_creator *c,
                              void **ptr,
                              size_t *size);

/**
 * Destroy a mapper creator before the last iteration of the copy.
 * This will free the device pointer currently built. If any branch were made
 * from this creator, it is the user responsibility to handle the branches and
 * finished branches pointers destruction separately. This function does not
 * check its input is valid.
 * @return AML_SUCCESS
 */
int aml_mapper_creator_abort(struct aml_mapper_creator *crtr);

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
 * means that the structure allocation must be split at this point. Moreover,
 * there will not be enough room in the creator buffers to fit current field.
 * In this case, the user has to use either `aml_mapper_creator_branch()` to
 * deepcopy the corresponding field or `aml_mapper_creator_connect()` to connect
 * an existing copy of the field. After one of these functions succeeds,
 * the user can resume calling this function.
 *
 * @param[in, out] c: A mapper creator representing the current state of a
 * structure copy from a source pointer to the host.
 * @return AML_SUCCESS on success to process this step.
 * @return -AML_EINVAL if the current field has the flag
 * `AML_MAPPER_FLAG_SPLIT` set. In that case, the next call should be
 * `aml_mapper_creator_branch()` or `aml_mapper_creator_connect()` instead.
 * @return -AML_EDOM there is no next field to copy. In that case,
 * the next call should be `aml_mapper_creator_finish()`.
 * @return Any error from dma_src_host arising from copying source pointer
 * to host.
 *
 * @see aml_mapper_creator_branch()
 * @see aml_mapper_creator_finish()
 */
int aml_mapper_creator_next(struct aml_mapper_creator *c);

/**
 * Connect an already instanciated structure to the structure being constructed.
 *
 * This function shall be called after `aml_mapper_creator_next()` returns
 * `-AML_EINVAL` with the same creator.
 * The structure to connect must match the mapper associated with the current
 * state of the creator. Moreover, this function expects that the mapper
 * of the creator current state has the flag `AML_MAPPER_FLAG_SPLIT` set.
 *
 * @param[in, out] c: A creator in a "split" state.
 * @param[in] ptr: The pointer to a constructed structure accurately described
 * by the mapper of the current state of the creator.
 * @return AML_SUCCESS on success, with the creator state advanced to the next
 * element to copy.
 * @return -AML_EDOM if there is nothing left to copy with the creator.
 * @return -AML_EINVAL if the creator is NULL, of if it does not have the flag
 * `AML_MAPPER_FLAG_SPLIT` set, or if `ptr` is NULL.
 * @return -AML_ENOMEM if there was not enough memory to move the creator state
 * to the next field to copy.
 */
int aml_mapper_creator_connect(struct aml_mapper_creator *c, void *ptr);

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
 * @return -AML_EDOM if there is nothing left to copy in the original
 * creator after this call. In that case, the next call on orginial creator
 * should be `aml_mapper_creator_finish()`.
 * @return Any error from `aml_mapper_creator_create()` if the creation of
 * the new creator failed.
 *
 * @see aml_mapper_creator_create()
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
