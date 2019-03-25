include_HEADERS=aml.h

include_aml_dmadir=$(includedir)/aml/dma
include_aml_dma_HEADERS = \
			      aml/dma/linux-seq.h \
			      aml/dma/linux-par.h

include_aml_scratchdir=$(includedir)/aml/scratch
include_aml_scratch_HEADERS = \
			      aml/scratch/seq.h \
			      aml/scratch/par.h


include_amlutilsdir=$(includedir)/aml/utils
include_amlutils_HEADERS = \
	aml/utils/vector.h \
	aml/utils/bitmap.h \
	aml/utils/version.h \
	aml/utils/error.h

if HAVE_HWLOC
include_amlutils_HEADERS+=\
	aml/utils/hwloc.h
endif

include_amlareadir=$(includedir)/aml/area
include_amlarea_HEADERS = \
	aml/area/linux.h
if HAVE_HWLOC
include_amlarea_HEADERS+=\
	aml/area/hwloc.h
endif
if HAVE_LINUX_NUMA
include_amlarea_HEADERS+=\
	aml/area/linux_numa.h
endif

include_amlscratchdir=$(includedir)/aml/scratch
include_amlscratch_HEADERS = \
	aml/scratch/double.h

include_amltilingdir=$(includedir)/aml/tiling
include_amltiling_HEADERS = \
	aml/tiling/tiling.h

noinst_HEADERS=\
	aml/area/area.h

