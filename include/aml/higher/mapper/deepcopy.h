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
 * @addtogroup aml_mapper
 *
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
typedef void *aml_mapped_ptrs;

struct aml_mapped_ptr {
	void *ptr;
	size_t size;
	struct aml_area *area;
};

void aml_mapped_ptr_destroy(void *ptr);

/**
 * Perform a deepcopy of a structure described with a `mapper` into an `area` of
 * choice.
 *
 * The structure to copy may be on host or on any device.
 * Regardless of whether the structure is on host or not, the structure will be
 * copied on host first and then to the `area` of choice. Therefore, this
 * function call may use as much memory as the structure to copy itself.
 *
 * Note that the structure to copy must have a tree topology, i.e no field
 * should reference a parent node in the structure, or reference the same
 * node as another field.
 *
 * This function requires that the pointer yielded in the desired area can be
 * safely offseted (not dereferenced) from host as long as the result pointer
 * is within the bounds of allocation. If the resulting pointer do not support
 * this property, then using this function is undefined.
 *
 * @param out[out]: A pointer where to store the allocated pointers of this
 * copy. This input must be used to cleanup the copy of the data structure
 * later.
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
 * @return A pointer to the copy of the structure on success.
 * @return NULL on error with aml_errno set as follow:
 * + -AML_EINVAL if `out`, `ptr`, `mapper`, `area`, `dma_host_dst`,
 * or `memcpy_host_dst` is NULL. This error might also be returned if one a
 * pointer of a field to copy in the source structure is NULL.
 * + Any error code created by area when to allocating data.
 * + Any error code from a failing dma engine copy.
 */
void *aml_mapper_deepcopy(aml_mapped_ptrs *out,
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
 * The area inside used to allocate the copy must still be valid when calling
 * this function.
 * @param data[in, out]: The structure returned as a result of a deepcopy.
 * @return AML_SUCCESS on success.
 * @return Any error code from `aml_area_munmap()` if it fails on a pointer.
 * In this case, the last pointer in `data` structure will be the problematic
 * one.
 */
int aml_mapper_deepfree(aml_mapped_ptrs data);

/**
 * @}
 **/

#ifdef __cplusplus
}
#endif
#endif // AML_MAPPER_DEEPCOPY_H
