include_HEADERS=aml.h

include_amlutilsdir=$(includedir)/aml/utils
include_amlutils_HEADERS = \
	aml/utils/vector.h \
	aml/utils/bitmap.h
if HAVE_HWLOC
include_amlutils_HEADERS+=\
	aml/utils/hwloc.h
endif

include_amlareadir=$(includedir)/aml/area
include_amlarea_HEADERS =
if HAVE_HWLOC
include_amlarea_HEADERS+=\
	aml/area/hwloc.h
endif

include_amlscratchdir=$(includedir)/aml/scratch
include_amlscratch_HEADERS = \
	aml/scratch/double.h

include_amltilingdir=$(includedir)/aml/tiling
include_amltiling_HEADERS = \
	aml/tiling/tiling.h

noinst_HEADERS=\
	aml/area/area.h

