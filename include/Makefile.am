include_HEADERS=aml.h

include_amlutilsdir=$(includedir)/aml/utils
include_amlutils_HEADERS = \
			   aml/utils/vector.h \
			   aml/utils/bitmap.h

if HAVE_HWLOC
include_amlutils_HEADERS+=\
	aml/utils/hwloc.h \
	aml/area/hwloc.h
endif
