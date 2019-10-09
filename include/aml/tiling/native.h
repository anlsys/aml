/*************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
**************************************************************************/

#ifndef AML_TILING_NATIVE_H
#define AML_TILING_NATIVE_H

/**
 * @defgroup aml_tiling_native "AML Tiling Internal API"
 * @brief API for internal management of tilings.
 *
 * @code
 * #include <aml/tiling/native.h>
 * @endcode
 * @{
 **/

struct aml_layout *aml_tiling_index_native(const struct aml_tiling *tiling,
					   const size_t *coords);

int aml_tiling_dims_native(const struct aml_tiling *tiling, size_t *dims);
/**
 * @}
 **/

#endif // AML_TILING_NATIVE_H
