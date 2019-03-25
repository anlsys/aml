#ifndef AML_ERROR_H
#define AML_ERROR_H

/**
 * Variable set by aml function calls. aml_errno should be checked after an aml function call
 * returning an error prior to any other aml call that may overwrite it.
 **/
extern int aml_errno;

/**
 * Get a string description of an aml error.
 * "errno": the aml error number.
 * Returns a on stack string with error description.
 **/
const char* aml_strerror(const int errno);

/**
 * Print error on standard error output.
 * "msg": A message to prepend to error message.
 **/
void aml_perror(const char * msg);

#define AML_SUCCESS       0 /* Generic value for success */
#define AML_FAILURE      -1 /* Generic value for failure */

/************************************ 
 * Area error codes  -2 .. -32 
 ************************************/

#define AML_AREA_EINVAL  -2 /* Invalid argument provided */
#define AML_AREA_ENOTSUP -3 /* Function not implemented for this type of area */
#define AML_AREA_ENOMEM  -4 /* Allocation failed */
#define AML_AREA_EDOM    -5 /* One arguent is out of allowed bounds */

/************************************ 
 * error bound
 ************************************/

#define AML_ERROR_MAX    -6 /* Last error */

#endif
