#include "aml/utils/error.h"

const char*
aml_strerror(const int errno)
{
	switch(errno){
	case AML_SUCCESS:
		return "aml success! If this is unexpected, check that this is called right after aml function returning an error.";
	case AML_FAILURE:
		return "aml function call failed (generic error).";
	case AML_AREA_EINVAL:		
		return "aml_area function called with invalid argument(s).";
	case AML_AREA_ENOTSUP:
		return "aml_area function is not implemented.";
	case AML_AREA_ENOMEM:
		return "Not enough memory to fulfill aml_area function call.";
	case AML_AREA_EDOM:
		return "An argument is out possible bounds for this function call.";
	default:
		return "Invalid aml error code.";
	}
}

void
aml_perror(const char * msg)
{
	fprintf(stderr, "%s:%s\n", msg, aml_strerror(aml_errno));
}

