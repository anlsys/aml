/*****************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ****************************************************************************/

#ifndef AML_DMA_HIP_H
#define AML_DMA_HIP_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup aml_dma_hip "AML DMA Hip"
 * @brief dma between devices and host.
 *
 * Hip dma is an implementation of aml dma to transfer data
 * between devices and between host and devices.
 *
 * @code
 * #include <aml/dma/hip.h>
 * @endcode
 * @see aml_dma
 * @{
 **/

#include <hip/hip_runtime_api.h>

//--- DMA Requests --------------------------------------------------------//

#define AML_DMA_HIP_REQUEST_STATUS_NONE 0
#define AML_DMA_HIP_REQUEST_STATUS_PENDING 1
#define AML_DMA_HIP_REQUEST_STATUS_DONE 2

/** Hip DMA request. Only need a status flag is needed. **/
struct aml_dma_hip_request {
	int status;
};

/**
 * AML dma hip request creation operator.
 * @return -AML_EINVAL if data, req, *req, dest or src is NULL.
 * @return -AML_ENOMEM if allocation failed.
 * @return AML_SUCCESS on success.
 **/
int aml_dma_hip_request_create(struct aml_dma_data *data,
                               struct aml_dma_request **req,
                               struct aml_layout *dest,
                               struct aml_layout *src,
                               aml_dma_operator op,
                               void *op_arg);

/**
 * AML dma hip request wait operator.
 * @return -AML_EINVAL if dma, req, *req is NULL or if data was does not
 * come from the dma used in request creation.
 * @return AML_SUCCESS on success.
 **/
int aml_dma_hip_request_wait(struct aml_dma_data *dma,
                             struct aml_dma_request **req);

/**
 * AML dma hip barrier operator.
 * @return AML_SUCCESS on success.
 **/
int aml_dma_hip_barrier(struct aml_dma_data *data);

/** AML dma hip request deletion operator **/
int aml_dma_hip_request_destroy(struct aml_dma_data *dma,
                                struct aml_dma_request **req);

//--- DMA -----------------------------------------------------------------//

/**
 * aml_dma data structure.
 * AML dma hip contains a single execution stream. When waiting
 * a request, the whole request stream is synchronized and all
 * the requests are waited.
 **/
struct aml_dma_hip_data {
	hipStream_t stream;
	enum hipMemcpyKind kind;
};

/** Default dma ops used at dma creation **/
extern struct aml_dma_ops aml_dma_hip_ops;

/**
 * Dma on stream 0 with `hipMemcpyDefault` copy kind.
 * Requires that the system supports unified virtual memory.
 */
extern struct aml_dma aml_dma_hip;

/**
 * Creation of a dma engine for hip backend.
 * @param dma: A pointer to set with a new allocated dma.
 * @param kind: The kind of transfer performed: host to device,
 * device to host, device to device, or host to host.
 * @see struct aml_dma_hip_data.
 * @return -AML_EINVAL if dma can't be set.
 * @return -AML_FAILURE if any hip backend call failed.
 * @return -AML_ENOMEM if allocation failed.
 * @return AML_SUCCESS on success.
 **/
int aml_dma_hip_create(struct aml_dma **dma, const enum hipMemcpyKind kind);

/** Destroy a created dma and set it to NULL **/
int aml_dma_hip_destroy(struct aml_dma **dma);

//--- DMA copy operators --------------------------------------------------//

/**
 * Embed a pair of devices in a void* to use as dma copy_operator argument
 * when copying from device to device.
 **/
#define AML_DMA_HIP_DEVICE_PAIR(src, dst)                                      \
	(void *)(((intptr_t)dst << 32) | ((intptr_t)src))

/**
 * Translate back a pair of device ids stored in `pair` (void*) into to
 * device id integers.
 **/
#define AML_DMA_HIP_DEVICE_FROM_PAIR(pair, src, dst)                           \
	src = dst = 0;                                                         \
	src = ((intptr_t)pair & 0xffffffff);                                   \
	dst = ((intptr_t)pair >> 32);

/**
 * Structure passed to `aml_dma_operator` `arg` argument by
 * the request created in `aml_dma_hip_request_create()`.
 * All `aml_dma_operator` implementations can expect to obtain
 * a pointer to this structure as `arg` argument. The pointer is
 * valid only for the lifetime of the `aml_dma_operator` call.
 */
struct aml_dma_hip_op_arg {
	// The calling dma data.
	struct aml_dma_hip_data *data;
	// The user extra dma_op arguments passed to
	// `aml_dma_hip_request_create()`.
	void *op_arg;
};

/**
 * Hip DMA operator implementation:
 * Use only with `aml_dma_hip_request_create()` or higher level
 * `aml_dma_async_copy_custom()`.
 * This copy operator is compatible only with:
 * - This dma hip implementation,
 * - Dense source and destination layouts of one dimension.
 * Make a flat copy of contiguous bytes in between two layout raw pointers.
 * The size of the byte stream is computed as the product of dimensions and
 * element size.
 * @param [in] dst: The destination layout of the copy.
 * @param [in] src: The source layout of the copy.
 * @param [in] arg: A pair of device ids obtained with
 * `AML_DMA_HIP_DEVICE_PAIR`.`op_arg` is used only if the dma used with
 * this operator is `hipMemcpyDeviceToDevice` kind of dma.
 * @return an AML error code.
 *
 * @see aml_layout_dense
 **/
int aml_dma_hip_copy_1D(struct aml_layout *dst,
                        const struct aml_layout *src,
                        void *arg);

/**
 * Hip DMA operator implementation:
 * Use only with `aml_dma_hip_request_create()` or higher level
 * `aml_dma_async_copy_custom()`.
 * This copy operator is compatible only with:
 * - This dma hip implementation (device to device is not supported),
 * - Flat source and destination pointers.
 * Make a flat asychronous copy of contiguous bytes between two raw
 * pointers. This dma operator casts input layout pointers into `void*` and
 * assumes these are contiguous set of bytes to copy from `src` to `dst`
 * in the linux `memcpy()` fashion with `hipMemcpyAsync()`.
 * @param[out] dst: The destination (`void*`) of the copy casted into a
 * `struct aml_layout *`.
 * @param[in] src: The source (`void*`) of the copy casted into a
 * `struct aml_layout *`.
 * @param[in] arg: The size (`size_t`) of the copy casted into a `void*`.
 * @return AML_SUCCESS
 */
int aml_dma_hip_memcpy_op(struct aml_layout *dst,
                          const struct aml_layout *src,
                          void *arg);

/**
 * @}
 **/

#ifdef __cplusplus
}
#endif

#endif // AML_LAYOUT_HIP_H
