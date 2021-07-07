/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#ifndef AML_UTILS_LINUX_H
#define AML_UTILS_LINUX_H

#ifdef __cplusplus
extern "C" {
#endif

#include "aml/dma/linux-par.h"
#include "aml/dma/linux-seq.h"

int aml_linux_init(void);
int aml_linux_finalize(void);

/**
 * @}
 **/

#ifdef __cplusplus
}
#endif
#endif // AML_UTILS_LINUX_H
