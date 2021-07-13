/*****************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ****************************************************************************/

#ifndef AML_DMA_ZE_H
#define AML_DMA_ZE_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup aml_dma_ze "AML DMA Ze"
 * @brief dma between devices and host using level zero.
 *
 * Ze dma is an implementation of aml dma to transfer data
 * between devices and host using level zero backend.
 * This dma implementation uses asynchronous immediate command queues on a
 * single target device.
 * @see
 *https://spec.oneapi.com/level-zero/latest/core/PROG.html#command-queues-and-command-lists
 *
 * @code
 * #include <aml/dma/ze.h>
 * @endcode
 * @see aml_dma
 * @{
 **/

#include <level_zero/ze_api.h>

//--- DMA -----------------------------------------------------------------//

/** Data structure of aml ze dma data. */
struct aml_dma_ze_data {
	// Logical context of this dma
	ze_context_handle_t context;
	// Target device of this dma
	ze_device_handle_t device;
	// Command list of this dma
	ze_command_list_handle_t command_list;
	// handle to events associated with this dma
	ze_event_pool_handle_t events;
	// The maximum of number of events that can be handled simultaneously in
	// the event pool.
	uint32_t event_pool_size;
	// Synchronization scope for events.
	ze_event_scope_flags_t event_flags;
};

/** Default dma ops used at dma creation **/
extern struct aml_dma_ops aml_dma_ze_ops;
/** Dma using first device of first driver **/
extern struct aml_dma *aml_dma_ze_default;

/**
 * Creation of a dma engine for ze backend. This dma engine can only
 * perform continuous and contiguous copies.
 * The underlying dma immediate command queue is set with:
 * + ZE_COMMAND_QUEUE_FLAG_EXPLICIT_ONLY:
 * "command queue should be optimized for submission to a single device engine"
 * + ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS:
 * "Device execution is scheduled and will complete in future"
 * + ZE_COMMAND_QUEUE_PRIORITY_NORMAL:
 * "[default] normal priority"
 *
 * @param dma: A pointer to set with a new allocated dma.
 * @param device: The target device that will be used as a dma engine.
 * @param pool_size: The maximum number of events that can be handled
 * simultaneously.
 * @see struct aml_dma_ze_data.
 * @return -AML_ENOMEM if allocation failed.
 * @return AML_SUCCESS on success.
 * @return Another aml error code translated from a `ze_result_t` that can
 * result from a ze resource creation (context, command queue, event pool).
 **/
int aml_dma_ze_create(struct aml_dma **dma,
                      ze_device_handle_t device,
                      const uint32_t pool_size);

/** Destroy a created dma and set it to NULL **/
int aml_dma_ze_destroy(struct aml_dma **dma);

//--- DMA request ---------------------------------------------------------//

/** aml ze request structure */
struct aml_dma_ze_request {
	// event handle
	ze_event_handle_t event;
};

/**
 * AML dma ze request creation operator.
 * @param data: The dma engine used to create request.
 * @param req: The pointer to the space where to allocate the request.
 * @param dest: The destination layout of the request.
 * @param src: The source layout of the request.
 * @param op: This dma implementation performs 1D copies only. Hence,
 * `op` argument is ignored.
 * @param op_arg: unused.
 * @return AML_SUCCESS on success.
 * @return -AML_ENOMEM if allocation failed.
 * @return Another aml error code translated from a `ze_result_t` that can
 * result from a event creation or copy submission.
 **/
int aml_dma_ze_request_create(struct aml_dma_data *data,
                              struct aml_dma_request **req,
                              struct aml_layout *dest,
                              struct aml_layout *src,
                              aml_dma_operator op,
                              void *op_arg);

/**
 * AML dma ze request wait operator.
 * @param dma: The dma engine used to create request.
 * @param req: The pointer to the request to wait.
 * @return AML_SUCCESS on success.
 * @return Another aml error code translated from a `ze_result_t` resulting
 * from event synchronization.
 **/
int aml_dma_ze_request_wait(struct aml_dma_data *dma,
                            struct aml_dma_request **req);

/**
 * AML dma ze request deletion operator.
 * @param dma: The dma engine used to create request.
 * @param req: The pointer to the request to free.
 * @return AML_SUCCESS on success.
 **/
int aml_dma_ze_request_destroy(struct aml_dma_data *dma,
                               struct aml_dma_request **req);

/**
 * Structure passed to `aml_dma_operator` `arg` argument by
 * the request created in `aml_dma_ze_request_create()`.
 * All `aml_dma_operator` implementations can expect to obtain
 * a pointer to this structure as `arg` argument. The pointer is
 * valid only for the lifetime of the `aml_dma_operator` call.
 */
struct aml_dma_ze_copy_args {
	// The calling dma data.
	struct aml_dma_ze_data *ze_data;
	// The pointer to the created request data.
	struct aml_dma_ze_request *ze_req;
	// Extra user args
	void *arg;
};

/**
 * AML dma copy operator.
 * This copy operator assumes a 1D layout of contiguous elements,
 * and submits a `zeCommandListAppendMemoryCopy()` command to the dma
 * command queue with input layouts raw pointers.
 * @param dst: The layout where to copy data. `dst` must be a 1 dimensional
 * layout of contiguous elements.
 * @param src: The layout from where to copy data. `src` must be a 1 dimensional
 * layout of contiguous elements.
 * @param arg: This argument is set by `aml_dma_ze_request_create()` call.
 * It contains a pointer to the following structure:
 * ```
 * struct aml_dma_ze_copy_args {
 * 	struct aml_dma_ze_data *ze_data;
 * 	struct aml_dma_ze_request *ze_req;
 * };
 * ```
 * where `ze_data` is the handle to the dma performing the operation
 * and `ze_req` is the handle to the resulting request containing the
 * instanciated but not initialized event to poll
 * on `aml_dma_ze_request_wait()`.
 * @return AML_SUCCESS on success.
 **/
int aml_dma_ze_copy_1D(struct aml_layout *dst,
                       const struct aml_layout *src,
                       void *arg);
/**
 * @}
 **/

#ifdef __cplusplus
}
#endif

#endif // AML_DMA_ZE_H
