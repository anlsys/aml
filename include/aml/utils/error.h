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
 * @defgroup aml_error "AML Error Management"
 * @brief AML Error Codes and Functions
 *
 * Error codes and error helper functions.
 * As is quite common in C code, error code values are defined in positive,
 * but are returned in negative.
 * @{
 **/

/**
 * Variable set by aml function calls. aml_errno should be checked after an aml
 * function call returning an error and prior to any other aml call that may
 * overwrite it.
 **/
extern int aml_errno;

/**
 * Get a string description of an aml error.
 * @param err: the aml error number.
 * Returns a static string describing the error.
 **/
const char *aml_strerror(const int err);

/**
 * Print error on standard error output.
 * "msg": A message to prepend to error message.
 **/
void aml_perror(const char *msg);

/** Generic value for success **/
#define AML_SUCCESS	0

/**
 * Generic value for failure
 * Usually when this is the returned value,
 * the function will detail another way to
 * inspect error.
 **/
#define AML_FAILURE	1

/**
 * Not enough memory was available
 * for fulfilling AML function call.
 **/
#define AML_ENOMEM	2

/**
 * One of the argument provided to
 * AML function call was invalid.
 **/
#define AML_EINVAL	3

/**
 * One of the arguments provided
 * to AML function call has out of bound
 * value.
 **/
#define AML_EDOM	4

/**
 * Invoked AML abstraction function is actually
 * not implemented for this particular version of
 * AML abstraction.
 **/
#define AML_ENOTSUP     5

/**
 * Invoked AML abstraction function is has failed
 * because the resource it works on was busy.
 **/
#define AML_EBUSY       6

/**
 * Max allowed value for errors.
 **/
#define AML_ERROR_MAX   7

/**
 * @}
 **/

#endif //AML_ERROR_H
