/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
*******************************************************************************/

#ifndef AML_ERROR_H
#define AML_ERROR_H

/**
 * Variable set by aml function calls. aml_errno should be checked after an aml
 * function call returning an error and prior to any other aml call that may
 * overwrite it.
 **/
extern int aml_errno;

/**
 * Get a string description of an aml error.
 * @param errno: the aml error number.
 * Returns a static string describing the error.
 **/
const char *aml_strerror(const int errno);

/**
 * Print error on standard error output.
 * "msg": A message to prepend to error message.
 **/
void aml_perror(const char *msg);

/**
 * Error codes.
 * As is quite common in C code, error code values are defined in positive,
 * but are returned in negative.
 */
#define AML_SUCCESS	0	/* Generic value for success */
#define AML_FAILURE	1	/* Generic value for failure */
#define AML_ENOMEM	2	/* No enough memory available. */
#define AML_EINVAL	3	/* Invalid argument provided */
#define AML_EDOM	4	/* value out of bound. */
#define AML_ENOTSUP	5	/* Operation not supported */
#define AML_ERROR_MAX	6	/* Max allowed value for errors. */

#endif
