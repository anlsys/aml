/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include "aml.h"

#include "aml/dma/hip.h"

/**
 * Callback on dma stream to update all requests status
 * When the stream is done with it work.
 **/
void aml_dma_hip_callback(hipStream_t stream, hipError_t status, void *userData)
{
	(void)stream;
	(void)status;
	struct aml_dma_hip_request *req;

	req = (struct aml_dma_hip_request *)userData;
	pthread_mutex_lock(&req->lock);
	req->status = AML_DMA_HIP_REQUEST_STATUS_DONE;
	pthread_mutex_unlock(&req->lock);
}

int aml_dma_hip_create(struct aml_dma **dma, const enum hipMemcpyKind kind)
{
	int err = AML_SUCCESS;

	struct aml_dma *hip_dma;
	struct aml_dma_hip_data *data;
	hipStream_t stream;

	// Argument check
	if (dma == NULL)
		return -AML_EINVAL;

	// Create a stream for this dma
	if (hipStreamCreateWithFlags(&stream, hipStreamNonBlocking) !=
	    hipSuccess)
		return -AML_FAILURE;

	// Create dma
	hip_dma = AML_INNER_MALLOC(struct aml_dma, struct aml_dma_hip_data);
	if (hip_dma == NULL) {
		err = -AML_EINVAL;
		goto error_with_stream;
	}

	// Set dma fields
	data = AML_INNER_MALLOC_GET_FIELD(hip_dma, 2, struct aml_dma,
	                                  struct aml_dma_hip_data);
	data->stream = stream;
	data->kind = kind;
	hip_dma->data = (struct aml_dma_data *)data;
	hip_dma->ops = &aml_dma_hip_ops;

	// Return success
	*dma = hip_dma;
	return AML_SUCCESS;

error_with_stream:
	hipStreamDestroy(stream);
	return err;
}

int aml_dma_hip_destroy(struct aml_dma **dma)
{
	struct aml_dma_hip_data *data;

	if (dma == NULL)
		return -AML_EINVAL;
	if (*dma == NULL)
		return AML_SUCCESS;

	data = (struct aml_dma_hip_data *)((*dma)->data);

	// Synchronize all requests.
	hipStreamSynchronize(data->stream);

	// Cleanup
	hipStreamDestroy(data->stream);
	free(*dma);
	*dma = NULL;
	return AML_SUCCESS;
}

int aml_dma_hip_request_create(struct aml_dma_data *data,
                               struct aml_dma_request **req,
                               struct aml_layout *dest,
                               struct aml_layout *src,
                               aml_dma_operator op,
                               void *op_arg)
{
	int err;
	struct aml_dma_hip_request *request = NULL;
	struct aml_dma_hip_data *dma_data = (struct aml_dma_hip_data *)data;

	// Set request
	if (req != NULL) {
		request = AML_INNER_MALLOC(struct aml_dma_hip_request);
		if (request == NULL)
			return -AML_ENOMEM;
		request->status = AML_DMA_HIP_REQUEST_STATUS_PENDING;
		pthread_mutex_init(&request->lock, NULL);
	}

	// Submit request to hip device
	struct aml_dma_hip_op_arg args = {
	        .data = dma_data,
	        .op_arg = op_arg,
	};
	err = op(dest, src, (void *)(&args));
	if (err != AML_SUCCESS) {
		pthread_mutex_destroy(&request->lock);
		free(request);
		return err;
	}

	// Also enqueue the callback to notify request is done.
	if (req != NULL) {
		if (hipStreamAddCallback(dma_data->stream, aml_dma_hip_callback,
		                         request, 0) != hipSuccess) {
			pthread_mutex_destroy(&request->lock);
			free(request);
			return -AML_FAILURE;
		}
		*req = (struct aml_dma_request *)request;
	}

	return AML_SUCCESS;
}

int aml_dma_hip_request_wait(struct aml_dma_data *data,
                             struct aml_dma_request **req)
{
	struct aml_dma_hip_data *dma_data;
	struct aml_dma_hip_request *dma_req;
	int status;

	if (req == NULL || *req == NULL)
		return -AML_EINVAL;

	dma_data = (struct aml_dma_hip_data *)(data);
	dma_req = (struct aml_dma_hip_request *)(*req);

	// If already done, do nothing
	pthread_mutex_lock(&dma_req->lock);
	status = dma_req->status;
	pthread_mutex_unlock(&dma_req->lock);
	if (status == AML_DMA_HIP_REQUEST_STATUS_DONE)
		goto exit_success;

	// Wait for the stream to finish and call its callback.
	hipStreamSynchronize(dma_data->stream);

	// If status is not updated, either callback failed or
	// the provided dma did not create the provided request.
	pthread_mutex_lock(&dma_req->lock);
	status = dma_req->status;
	pthread_mutex_unlock(&dma_req->lock);
	if (status != AML_DMA_HIP_REQUEST_STATUS_DONE)
		return -AML_EINVAL;

exit_success:
	pthread_mutex_destroy(&dma_req->lock);
	free(dma_req);
	*req = NULL;
	return AML_SUCCESS;
}

int aml_dma_hip_barrier(struct aml_dma_data *data)
{
	struct aml_dma_hip_data *dma_data;

	dma_data = (struct aml_dma_hip_data *)(data);

	// Wait for the stream to finish and call its callback.
	hipStreamSynchronize(dma_data->stream);
	return AML_SUCCESS;
}

int aml_dma_hip_request_destroy(struct aml_dma_data *data,
                                struct aml_dma_request **req)
{
	struct aml_dma_hip_data *dma_data;
	struct aml_dma_hip_request *dma_req;
	int status;

	if (req == NULL || *req == NULL)
		return -AML_EINVAL;

	dma_data = (struct aml_dma_hip_data *)(data);
	dma_req = (struct aml_dma_hip_request *)(*req);

	// If the request status is not done, wait for it to be done.
	// This way, the stream callback will not update a deleted request.
	pthread_mutex_lock(&dma_req->lock);
	status = dma_req->status;
	pthread_mutex_unlock(&dma_req->lock);
	if (status != AML_DMA_HIP_REQUEST_STATUS_DONE)
		hipStreamSynchronize(dma_data->stream);

	// If status is not updated, either callback failed or
	// the provided dma did not create the provided request.
	pthread_mutex_lock(&dma_req->lock);
	status = dma_req->status;
	pthread_mutex_unlock(&dma_req->lock);
	if (status != AML_DMA_HIP_REQUEST_STATUS_DONE)
		return -AML_EINVAL;

	// Cleanup
	pthread_mutex_destroy(&dma_req->lock);
	free(dma_req);
	*req = NULL;
	return AML_SUCCESS;
}

int aml_dma_hip_copy_1D(struct aml_layout *dst,
                        const struct aml_layout *src,
                        void *arg)
{
	int err;
	int src_device;
	int dst_device;

	const void *src_ptr = aml_layout_rawptr(src);
	void *dst_ptr = aml_layout_rawptr(dst);
	struct aml_dma_hip_op_arg *op_arg = (struct aml_dma_hip_op_arg *)arg;
	size_t n = 0;
	size_t size = 0;

	err = aml_layout_dims(src, &n);
	if (err != AML_SUCCESS)
		return err;
	size = aml_layout_element_size(src) * n;

	if (op_arg->data->kind == hipMemcpyDeviceToDevice) {
		AML_DMA_HIP_DEVICE_FROM_PAIR(op_arg->op_arg, src_device,
		                             dst_device);
		if (hipMemcpyPeerAsync(dst_ptr, dst_device, src_ptr, src_device,
		                       size,
		                       op_arg->data->stream) != hipSuccess)
			return -AML_FAILURE;
	} else {
		if (hipMemcpyAsync(dst_ptr, src_ptr, size, op_arg->data->kind,
		                   op_arg->data->stream) != hipSuccess)
			return -AML_FAILURE;
	}
	return AML_SUCCESS;
}

int aml_dma_hip_memcpy_op(struct aml_layout *dst,
                          const struct aml_layout *src,
                          void *arg)
{
	struct aml_dma_hip_op_arg *op_arg = (struct aml_dma_hip_op_arg *)arg;
	size_t size = (size_t)op_arg->op_arg;

	if (op_arg->data->kind == hipMemcpyDeviceToDevice)
		return -AML_ENOTSUP;
	else {
		if (hipMemcpyAsync(dst, src, size, op_arg->data->kind,
		                   op_arg->data->stream) != hipSuccess)
			return -AML_FAILURE;
	}
	return AML_SUCCESS;
}

/** Default dma ops **/
struct aml_dma_ops aml_dma_hip_ops = {
        .create_request = aml_dma_hip_request_create,
        .destroy_request = aml_dma_hip_request_destroy,
        .wait_request = aml_dma_hip_request_wait,
        .barrier = aml_dma_hip_barrier,
};

struct aml_dma_hip_data aml_dma_hip_data = {
        .stream = 0,
        .kind = hipMemcpyDefault,
};

struct aml_dma aml_dma_hip = {
        .ops = &aml_dma_hip_ops,
        .data = (struct aml_dma_data *)(&aml_dma_hip_data),
};
