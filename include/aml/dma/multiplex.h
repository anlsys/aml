/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#ifndef AML_DMA_MULTIPLEX_H
#define AML_DMA_MULTIPLEX_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup aml_dma_multiplex "AML DMA Multiplex"
 * @brief AML DMA engine implementation that multiplexes over existing dmas,
 * using a weighted distribution.
 *
 * @code
 * #include <aml/dma/multiplex.h>
 * @endcode

 * @see aml_async
 * @{
 **/

//------------------------------------------------------------------------------
// User API
//------------------------------------------------------------------------------

/**
 * Create a dma engine with a custom amount of dmas.
 *
 * @param[out] dma: A pointer where to allocate the dma engine.
 * @param[in] num: The number of dmas in the dmas array.
 * @param[in] dmas: The dmas to multiplex over
 * @param[in] weights The weights of each dmas.
 *
 * @return -AML_ENOMEM on error, exclusively caused when being out of memory.
 * @return AML_SUCCESS on success. On success, the created dma must be destroyed
 * with `aml_dma_multiplex_destroy()`.
 */
int aml_dma_multiplex_create(struct aml_dma **dma,
                             const size_t num,
                             const struct aml_dma **dmas,
                             const size_t *weights);

/**
 * Delete a multiplex dma created with `aml_dma_multiplex_create()`.
 *
 * @param[in, out] dma: A pointer where the dma engine has been allocated.
 * The pointer content is set to NULL after deallocation.
 * @return AML_SUCCESS.
 */
int aml_dma_multiplex_destroy(struct aml_dma **dma);

//------------------------------------------------------------------------------
// Internals API
//------------------------------------------------------------------------------

/** The dma data is an array of dmas and weitghts, plus info on which
 * dma to use next */
struct aml_dma_multiplex_data {
	size_t count;
	size_t index;
	size_t round;
	struct aml_dma **dmas;
	size_t *weights;
};

/** The methods table of multiplex dma. */
extern struct aml_dma_ops aml_dma_multiplex_ops;

/**
 * Request Flag of requests created but not returned to user that need
 * to be destroyed.
 */
#define AML_DMA_MULTIPLEX_REQUEST_FLAGS_OWNED 0x1

/**
 * Request Flag turned on when request is finished.
 */
#define AML_DMA_MULTIPLEX_REQUEST_FLAGS_DONE 0x2

/** The dma request implementation: number of underlying requests,
 * and their dmas
 */
struct aml_dma_multiplex_request {
	size_t count;
	struct aml_dma_request **reqs;
	struct aml_dma **dmas;
};

/**
 * User-level structure for operator arguments.
 * Will use the operator and the arg depending on the dma used.
 *
 * Arrays must be the same size as the dma array during create.
 */
struct aml_dma_multiplex_request_args {
	aml_dma_operator *ops;
	void **op_args;
};

/**
 * The multiplex dma `create_request()` operator implementation.
 * Creates a pointer `struct aml_dma_multiplex_request` stored in `req`.
 */
int aml_dma_multiplex_request_create(struct aml_dma_data *data,
                                     struct aml_dma_request **req,
                                     struct aml_layout *dest,
                                     struct aml_layout *src,
                                     aml_dma_operator op,
                                     void *op_arg);

/**
 * The multiplex dma `wait_request()` operator implementation.
 *
 * @param[in] dma: The dma engine where request has been posted.
 * @param[in] req: A pointer to a `struct aml_dma_multiplex_request`.
 */
int aml_dma_multiplex_request_wait(struct aml_dma_data *dma,
                                   struct aml_dma_request **req);

/**
 * The multiplex dma `barrier()` operator implementation.
 * @return The first failing request error code on error.
 * Remaining requests are not waited.
 * @return AML_SUCCESS on success.
 */
int aml_dma_multiplex_barrier(struct aml_dma_data *dma);

/**
 * The multiplex dma `destroy_request()` operator implementation.
 *
 * @param[in] dma: unused.
 * @param[in] req: A pointer to a `struct aml_dma_multiplex_request`.
 * The pointer is set to NULL.
 */
int aml_dma_multiplex_request_destroy(struct aml_dma_data *dma,
                                      struct aml_dma_request **req);

struct aml_dma_multiplex_copy_args {
	struct aml_dma_multiplex_data *m_data;
	struct aml_dma_multiplex_request **m_req;
	struct aml_dma_multiplex_request_args *args;
};

/**
 * multiplex DMA operator implementation:
 * Use only with `aml_dma_multiplex_request_create()` or higher level
 * `aml_dma_async_copy_custom()`.
 * This copy operator is compatible only with:
 * - This dma multiplex implementation.
 *
 * Send a single request to a dma, using the operator and arg defined for that
 * dma in the request_args parameter.
 *
 * @param[out] dst: The destination dense layout.
 * @param[in] src: The source dense layout.
 * @param[in] arg: pointer to struct aml_dma_multiplex_request_args.
 *
 * @see aml_layout_dense
 */
int aml_dma_multiplex_copy_single(struct aml_layout *dst,
                                  const struct aml_layout *src,
                                  void *arg);

/**
 * @}
 **/

#ifdef __cplusplus
}
#endif

#endif // AML_DMA_MULTIPLEX_H
