/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *******************************************************************************/

#include "aml.h"

/**
 * Generic test function.
 * This test allocates a witness buffer on host, a test buffer on host
 * and a transfer buffer on area. The test and witness buffers are
 * initialized with different values. Then the dma is used to copy
 * witness buffer in area buffer and area buffer back into test buffer.
 * Finally witness and test buffer are tested for bitwise equality.
 *
 * @param[in, out] area: The area used to allocate the area buffer.
 * @param[in] area_opts: The area options
 * @param[in, out] dma: The dma engine moving data back and forth between
 * host and area. The dma must be a bidirectional dma.
 * @param[in] memcpy_op: A dma operator performing a simple memcpy beetween
 * to pointers of the same size. The dma operator must reinterpreting
 * arguments as follow:
 * - aml_layout *dst -> void * (raw pointer to data)
 * - aml_layout *src -> void * (raw pointer to data)
 * - void *arg -> struct (The structure passed to op by the dma with the
 * pointers size as user argument. See dma implementations for more details.)
 * host and area.
 */
void test_dma_memcpy(struct aml_area *area,
                     struct aml_area_mmap_options *area_opts,
                     struct aml_dma *dma,
                     aml_dma_operator memcpy_op);

/**
 * Generic test function for barrier primitive.
 * This test allocates a set of witness buffer on host,
 * a set test buffer on host and a transfer buffer on area.
 * The test and witness buffers are initialized with different values.
 * Then the dma is used to copy witness buffers in area buffers and area
 * buffers back into test buffers. Copies are batched when going to the area
 * and batched when going back to the host and waited with a barrier.
 * Finally witness and test buffers are tested for bitwise equality.
 *
 * @param[in, out] area: The area used to allocate the area buffer.
 * @param[in] area_opts: The area options
 * @param[in, out] dma: The dma engine moving data back and forth between
 * host and area. The dma must be a bidirectional dma.
 * @param[in] memcpy_op: A dma operator performing a simple memcpy beetween
 * to pointers of the same size. The dma operator must reinterpreting
 * arguments as follow:
 * - aml_layout *dst -> void * (raw pointer to data)
 * - aml_layout *src -> void * (raw pointer to data)
 * - void *arg -> struct (The structure passed to op by the dma with the
 * pointers size as user argument. See dma implementations for more details.)
 * host and area.
 */
void test_dma_barrier(struct aml_area *area,
                      struct aml_area_mmap_options *area_opts,
                      struct aml_dma *dma,
                      aml_dma_operator memcpy_op);
