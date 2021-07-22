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

#include "aml/dma/cuda.h"

/**
 * Callback on dma stream to update all requests status
 * When the stream is done with it work.
 **/
void aml_dma_cuda_callback(void *userData)
{
	struct aml_dma_cuda_request *req;

	req = (struct aml_dma_cuda_request *)userData;
	req->status = AML_DMA_CUDA_REQUEST_STATUS_DONE;
}

int aml_dma_cuda_create(struct aml_dma **dma, const enum cudaMemcpyKind kind)
{
	int err = AML_SUCCESS;

	struct aml_dma *cuda_dma;
	struct aml_dma_cuda_data *data;
	cudaStream_t stream;

	// Argument check
	if (dma == NULL)
		return -AML_EINVAL;

	// Create a stream for this dma
	if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) !=
	    cudaSuccess)
		return -AML_FAILURE;

	// Create dma
	cuda_dma = AML_INNER_MALLOC(struct aml_dma, struct aml_dma_cuda_data);
	if (cuda_dma == NULL) {
		err = -AML_EINVAL;
		goto error_with_stream;
	}

	// Set dma fields
	data = AML_INNER_MALLOC_GET_FIELD(cuda_dma, 2, struct aml_dma,
	                                  struct aml_dma_cuda_data);
	data->stream = stream;
	data->kind = kind;
	cuda_dma->data = (struct aml_dma_data *)data;
	cuda_dma->ops = &aml_dma_cuda_ops;

	// Return success
	*dma = cuda_dma;
	return AML_SUCCESS;

error_with_stream:
	cudaStreamDestroy(stream);
	return err;
}

int aml_dma_cuda_destroy(struct aml_dma **dma)
{
	struct aml_dma_cuda_data *data;

	if (dma == NULL)
		return -AML_EINVAL;
	if (*dma == NULL)
		return AML_SUCCESS;

	data = (struct aml_dma_cuda_data *)((*dma)->data);

	// Synchronize all requests.
	cudaStreamSynchronize(data->stream);

	// Cleanup
	cudaStreamDestroy(data->stream);
	free(*dma);
	*dma = NULL;
	return AML_SUCCESS;
}

int aml_dma_cuda_request_create(struct aml_dma_data *data,
                                struct aml_dma_request **req,
                                struct aml_layout *dest,
                                struct aml_layout *src,
                                aml_dma_operator op,
                                void *op_arg)
{
	int err;
	struct aml_dma_cuda_request *request = NULL;
	struct aml_dma_cuda_data *dma_data = (struct aml_dma_cuda_data *)data;

	// Set request
	if (req != NULL) {
		request = AML_INNER_MALLOC(struct aml_dma_cuda_request);
		if (request == NULL)
			return -AML_ENOMEM;
		request->status = AML_DMA_CUDA_REQUEST_STATUS_PENDING;
	}

	// Submit request to cuda device
	struct aml_dma_cuda_op_arg args = {
	        .data = dma_data,
	        .op_arg = op_arg,
	};
	err = op(dest, src, (void *)(&args));
	if (err != AML_SUCCESS) {
		free(request);
		return err;
	}

	// Also enqueue the callback to notify request is done.
	if (req != NULL) {
		if (cudaLaunchHostFunc(dma_data->stream, aml_dma_cuda_callback,
		                       request) != cudaSuccess) {
			free(request);
			return -AML_FAILURE;
		}
		*req = (struct aml_dma_request *)request;
	}

	return AML_SUCCESS;
}

int aml_dma_cuda_request_wait(struct aml_dma_data *data,
                              struct aml_dma_request **req)
{
	struct aml_dma_cuda_data *dma_data;
	struct aml_dma_cuda_request *dma_req;

	if (req == NULL || *req == NULL)
		return -AML_EINVAL;

	dma_data = (struct aml_dma_cuda_data *)(data);
	dma_req = (struct aml_dma_cuda_request *)(*req);

	// If already done, do nothing
	if (dma_req->status == AML_DMA_CUDA_REQUEST_STATUS_DONE)
		goto exit_success;

	// Wait for the stream to finish and call its callback.
	cudaStreamSynchronize(dma_data->stream);

	// If status is not updated, either callback failed or
	// the provided dma did not create the provided request.
	if (dma_req->status != AML_DMA_CUDA_REQUEST_STATUS_DONE)
		return -AML_EINVAL;

exit_success:
	free(dma_req);
	*req = NULL;
	return AML_SUCCESS;
}

int aml_dma_cuda_barrier(struct aml_dma_data *data)
{
	struct aml_dma_cuda_data *dma_data;

	dma_data = (struct aml_dma_cuda_data *)(data);

	// Wait for the stream to finish and call its callback.
	cudaStreamSynchronize(dma_data->stream);
	return AML_SUCCESS;
}

int aml_dma_cuda_request_destroy(struct aml_dma_data *data,
                                 struct aml_dma_request **req)
{
	struct aml_dma_cuda_data *dma_data;
	struct aml_dma_cuda_request *dma_req;

	if (req == NULL || *req == NULL)
		return -AML_EINVAL;

	dma_data = (struct aml_dma_cuda_data *)(data);
	dma_req = (struct aml_dma_cuda_request *)(*req);

	// If the request status is not done, wait for it to be done.
	// This way, the stream callback will not update a deleted request.
	if (dma_req->status != AML_DMA_CUDA_REQUEST_STATUS_DONE)
		cudaStreamSynchronize(dma_data->stream);

	// If status is not updated, either callback failed or
	// the provided dma did not create the provided request.
	if (dma_req->status != AML_DMA_CUDA_REQUEST_STATUS_DONE)
		return -AML_EINVAL;

	// Cleanup
	free(dma_req);
	*req = NULL;
	return AML_SUCCESS;
}

int aml_dma_cuda_copy_1D(struct aml_layout *dst,
                         const struct aml_layout *src,
                         void *arg)
{
	int err;
	int src_device;
	int dst_device;

	const void *src_ptr = aml_layout_rawptr(src);
	void *dst_ptr = aml_layout_rawptr(dst);
	struct aml_dma_cuda_op_arg *op_arg = (struct aml_dma_cuda_op_arg *)arg;
	size_t n = 0;
	size_t size = 0;

	err = aml_layout_dims(src, &n);
	if (err != AML_SUCCESS)
		return err;
	size = aml_layout_element_size(src) * n;

	if (op_arg->data->kind == cudaMemcpyDeviceToDevice) {
		AML_DMA_CUDA_DEVICE_FROM_PAIR(op_arg->op_arg, src_device,
		                              dst_device);
		if (cudaMemcpyPeerAsync(dst_ptr, dst_device, src_ptr,
		                        src_device, size,
		                        op_arg->data->stream) != cudaSuccess)
			return -AML_FAILURE;
	} else {
		if (cudaMemcpyAsync(dst_ptr, src_ptr, size, op_arg->data->kind,
		                    op_arg->data->stream) != cudaSuccess)
			return -AML_FAILURE;
	}
	return AML_SUCCESS;
}

int aml_memcpy_cuda(struct aml_layout *dst,
                    const struct aml_layout *src,
                    void *arg)
{
	struct aml_dma_cuda_op_arg *op_arg = (struct aml_dma_cuda_op_arg *)arg;
	size_t size = (size_t)op_arg->op_arg;

	if (op_arg->data->kind == cudaMemcpyDeviceToDevice)
		return -AML_ENOTSUP;
	else {
		if (cudaMemcpyAsync(dst, src, size, op_arg->data->kind,
		                    op_arg->data->stream) != cudaSuccess)
			return -AML_FAILURE;
	}
	return AML_SUCCESS;
}

/** Default dma ops **/
struct aml_dma_ops aml_dma_cuda_ops = {
        .create_request = aml_dma_cuda_request_create,
        .destroy_request = aml_dma_cuda_request_destroy,
        .wait_request = aml_dma_cuda_request_wait,
        .barrier = aml_dma_cuda_barrier,
};

struct aml_dma_cuda_data aml_dma_cuda_data = {
        .stream = 0,
        .kind = cudaMemcpyDefault,
};

struct aml_dma aml_dma_cuda = {
        .ops = &aml_dma_cuda_ops,
        .data = (struct aml_dma_data *)(&aml_dma_cuda_data),
};
