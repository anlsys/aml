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

#include "aml/higher/replicaset.h"

#ifndef TEST_REPLICASET_H
#define TEST_REPLICASET_H 1

void test_replicaset(struct aml_replicaset *replicaset,
                     const void *src,
                     int (*comp)(const void *, const void *, size_t));

#endif
