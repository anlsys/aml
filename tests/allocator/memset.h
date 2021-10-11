/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *******************************************************************************/

typedef int (*aml_memset_fn)(void *ptr, int val, size_t size);

int aml_dummy_memset(void *ptr, int val, size_t size);

int aml_linux_memset(void *ptr, int val, size_t size);

int aml_cuda_memset(void *ptr, int val, size_t size);

int aml_ze_memset(void *ptr, int val, size_t size);
