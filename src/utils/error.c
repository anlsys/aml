#include <stdio.h>
#include "aml/utils/error.h"

static const char* aml_error_strings[-AML_ERROR_MAX] = {
	[AML_SUCCESS]       = "aml success! If this is unexpected, check that this is called right after aml function returning an error.",
	[-AML_FAILURE]      = "aml function call failed (generic error).",
	[-AML_AREA_EINVAL]  = "aml_area function called with invalid argument(s).",
	[-AML_AREA_ENOTSUP] = "aml_area function is not implemented.",
	[-AML_AREA_ENOMEM]  = "Not enough memory to fulfill aml_area function call.",
	[-AML_AREA_EDOM]    = "An argument is out possible bounds for this function call.",
};		

const char*
aml_strerror(const int errno)
{
	if( errno < AML_ERROR_MAX || errno > 0 || aml_error_strings[errno] == NULL )
		return "Invalid aml error code.";
	return aml_error_strings[errno];
}

void
aml_perror(const char * msg)
{
	fprintf(stderr, "%s:%s\n", msg, aml_strerror(aml_errno));
}

