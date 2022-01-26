/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#ifndef AML_MAPPER_DEEPCOPY_H
#define AML_MAPPER_DEEPCOPY_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup aml_deepcopy "AML Deep-copy"
 * @brief Deep copy of a hierarchical structure.
 * @{
 **/

/**
 * Data obtained as a result of a deepcopy operation used to free
 * allocated pointers.
 *
 * This struct is implemented as a UT_array where elements are
 * mapped pointers, with their size and the area which mapped them.
 * The first element of this array it the pointer to the root of the
 * data structure, i.e the deepcopied structure.
 */
typedef void *aml_deepcopy_data;

/** Get the pointer to the copy of the data structure deep copied. */
void *aml_deepcopy_ptr(aml_deepcopy_data data);

/**
 * Perform a deepcopy of a structure described with a `mapper` into an `area` of
 * choice.
 *
 * The structure to copy may be on host or on any device.
 * Regardless of whether the structure is on host or not, the structure will be
 * copied on host first and then to the `area` of choice. Therefore, this
 * function call may use as much memory as the structure to copy itself.
 *
 * This function requires that the pointer yielded in the desired area can be
 * safely offseted (not dereferenced) from host as long as the result pointer
 * is within the bounds of allocation. If the resulting pointer do not support
 * this property, then using this function is undefined.
 *
 * @param out[out]: A pointer where to store the allocated pointers of this
 * copy. The first allocated pointer is the pointer to the copied structure.
 * @see aml_deepcopy_ptr()
 * This structure is also used later to free the copy created during this call.
 * @see aml_mapper_deepfree()
 * @param ptr[in]: The pointer to the structure to copy.
 * @param mapper[in]: The mapper describing the structure to copy.
 * @param area[in]: The area where to allocate the copy.
 * @param dma_src_host[in]: A dma engine to copy from the area of the source
 * structure to copy to the host. This can be NULL if the source pointer is
 * already a host pointer.
 * @param dma_host_dst[in]: A dma engine to copy from the host to the area
 * where the source structure is copied.
 * @param memcpy_src_host[in]: The memcpy operator associated with
 * `dma_src_host`. It can be NULL if `dma_src_host` is also NULL.
 * @param memcpy_host_dst[in]: The memcpy operator associated with
 * `dma_host_dst`.
 * @return AML_SUCCESS on success. On success `out` will contain the pointer
 * to the copied data structure. `out` and the copied data structure can later
 * bee freed with `aml_mapper_deepfree()`.
 * @return -AML_EINVAL if `out`, `ptr`, `mapper`, `area`, `dma_host_dst`,
 * or `memcpy_host_dst` is NULL. This error might also be returned if one a
 * pointer of a field to copy in the source structure is NULL.
 * @return Any error code from `aml_errno` if area fails to allocate data..
 * @return Any error code from a failing dma engine copy.
 */
int aml_mapper_deepcopy(aml_deepcopy_data *out,
                        void *ptr,
                        struct aml_mapper *mapper,
                        struct aml_area *area,
                        struct aml_area_mmap_options *area_opts,
                        struct aml_dma *dma_src_host,
                        struct aml_dma *dma_host_dst,
                        aml_dma_operator memcpy_src_host,
                        aml_dma_operator memcpy_host_dst);

/**
 * Release resources allocated with `aml_mapper_deepcopy()`.
 * The area inside `data` must still be valid when calling this function.
 * @param data[in, out]: The structure returned as a result of a deepcopy.
 * @return AML_SUCCESS on success.
 * @return Any error code from `aml_area_munmap()` if it fails on a pointer.
 * In this case, the last pointer in `data` structure will be the problematic
 * one.
 */
int aml_mapper_deepfree(aml_deepcopy_data data);

/**
 * @}
 **/

#ifdef __cplusplus
}
#endif
#endif // AML_MAPPER_DEEPCOPY_H
