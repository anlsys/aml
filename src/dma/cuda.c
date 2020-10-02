/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include "aml.h"

#include "aml/dma/cuda.h"
#include "aml/layout/cuda.h"

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
	struct aml_dma_cuda_request *request;
	struct aml_dma_cuda_data *dma_data;

	(void)op_arg;

	// Check input
	if (data == NULL || req == NULL || dest == NULL || src == NULL)
		return -AML_EINVAL;
	dma_data = (struct aml_dma_cuda_data *)data;

	// Set request
	request = AML_INNER_MALLOC(struct aml_dma_cuda_request);
	if (request == NULL)
		return -AML_ENOMEM;
	request->status = AML_DMA_CUDA_REQUEST_STATUS_PENDING;

	// Submit request to cuda device
	err = op(dest, src, (void *)(dma_data));
	if (err != AML_SUCCESS) {
		free(request);
		return err;
	}

	// Also enqueue the callback to notfiy request is done.
	if (cudaLaunchHostFunc(dma_data->stream, aml_dma_cuda_callback,
	                       request) != cudaSuccess) {
		free(request);
		return -AML_FAILURE;
	}

	*req = (struct aml_dma_request *)request;
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
		return AML_SUCCESS;

	// Wait for the stream to finish and call its callback.
	cudaStreamSynchronize(dma_data->stream);

	// If status is not updated, either callback failed or
	// the provided dma did not create the provided request.
	if (dma_req->status != AML_DMA_CUDA_REQUEST_STATUS_DONE)
		return -AML_EINVAL;

	aml_dma_cuda_request_destroy(data, req);
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
	const void *src_ptr = aml_layout_rawptr(src);
	void *dst_ptr = aml_layout_rawptr(dst);
	struct aml_dma_cuda_data *dma_data = (struct aml_dma_cuda_data *)arg;
	const struct aml_layout_cuda_data *cu_src =
	        (struct aml_layout_cuda_data *)(src->data);
	struct aml_layout_cuda_data *cu_dst =
	        (struct aml_layout_cuda_data *)(dst->data);
	size_t n = 0;
	size_t size = 0;

	err = aml_layout_dims(src, &n);
	if (err != AML_SUCCESS)
		return err;
	size = aml_layout_element_size(src) * n;

	if (dma_data->kind == cudaMemcpyHostToDevice ||
	    dma_data->kind == cudaMemcpyDeviceToHost) {
		if (cudaMemcpyAsync(dst_ptr, src_ptr, size, dma_data->kind,
		                    dma_data->stream) != cudaSuccess)
			return -AML_FAILURE;
	} else if (dma_data->kind == cudaMemcpyDeviceToDevice) {
		if (cudaMemcpyPeerAsync(dst_ptr, cu_dst->device, src_ptr,
		                        cu_src->device, size,
		                        dma_data->stream) != cudaSuccess)
			return -AML_FAILURE;
	} else
		memcpy(dst_ptr, src_ptr, size);
	return AML_SUCCESS;
}

/** Default dma ops **/
struct aml_dma_ops aml_dma_cuda_ops = {
        .create_request = aml_dma_cuda_request_create,
        .destroy_request = aml_dma_cuda_request_destroy,
        .wait_request = aml_dma_cuda_request_wait,
};

struct aml_dma_cuda_data aml_dma_cuda_data_host_to_device = {
        .stream = 0,
        .kind = cudaMemcpyHostToDevice,
};

struct aml_dma aml_dma_cuda_host_to_device = {
        .ops = &aml_dma_cuda_ops,
        .data = (struct aml_dma_data *)(&aml_dma_cuda_data_host_to_device),
};

struct aml_dma_cuda_data aml_dma_cuda_data_device_to_host = {
        .stream = 0,
        .kind = cudaMemcpyDeviceToHost,
};

struct aml_dma aml_dma_cuda_device_to_host = {
        .ops = &aml_dma_cuda_ops,
        .data = (struct aml_dma_data *)(&aml_dma_cuda_data_device_to_host),
};

struct aml_dma_cuda_data aml_dma_cuda_data_device_to_device = {
        .stream = 0,
        .kind = cudaMemcpyDeviceToDevice,
};

struct aml_dma aml_dma_cuda_device_to_device = {
        .ops = &aml_dma_cuda_ops,
        .data = (struct aml_dma_data *)(&aml_dma_cuda_data_device_to_device),
};
