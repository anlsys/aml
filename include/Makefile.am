include_HEADERS=aml.h

include_aml_areadir=$(includedir)/aml/area
include_aml_area_HEADERS = aml/area/linux.h

include_aml_higherdir=$(includedir)/aml/higher
include_aml_higher_HEADERS = \
			      aml/higher/mapper.h \
			      aml/higher/replicaset.h \
			      aml/higher/allocator.h \
			      aml/higher/allocator/area.h \
			      aml/higher/allocator/buddy.h \
			      aml/higher/allocator/sized.h

include_aml_mapperdir=$(includedir)/aml/mapper
include_aml_mapper_HEADERS = \
			      aml/higher/mapper/visitor.h \
			      aml/higher/mapper/creator.h \
			      aml/higher/mapper/deepcopy.h \
			      aml/higher/mapper/replicaset.h

include_aml_replicasetdir=$(includedir)/aml/higher/replicaset
include_aml_replicaset_HEADERS = 

include_aml_layoutdir=$(includedir)/aml/layout
include_aml_layout_HEADERS = \
			      aml/layout/native.h \
			      aml/layout/dense.h \
			      aml/layout/pad.h \
			      aml/layout/reshape.h

include_aml_dmadir=$(includedir)/aml/dma
include_aml_dma_HEADERS = aml/dma/linux.h \
			  aml/dma/multiplex.h

include_aml_tilingdir=$(includedir)/aml/tiling
include_aml_tiling_HEADERS = \
			     aml/tiling/native.h \
			     aml/tiling/pad.h \
			     aml/tiling/resize.h


include_amlutilsdir=$(includedir)/aml/utils
include_amlutils_HEADERS = \
			   aml/utils/bitmap.h \
			   aml/utils/error.h \
			   aml/utils/inner-malloc.h \
			   aml/utils/vector.h \
			   aml/utils/queue.h \
			   aml/utils/async.h \
			   aml/utils/version.h \
			   aml/utils/macros.h \
			   aml/utils/features.h

include_aml_backenddir=$(includedir)/aml/utils/backend
include_aml_backend_HEADERS = aml/utils/backend/linux.h

if HAVE_HWLOC
include_aml_area_HEADERS+= aml/area/hwloc.h
include_aml_replicaset_HEADERS += aml/higher/replicaset/hwloc.h
include_aml_backend_HEADERS += aml/utils/backend/hwloc.h
endif

if HAVE_CUDA
include_aml_area_HEADERS += aml/area/cuda.h
include_aml_dma_HEADERS += aml/dma/cuda.h
endif

if HAVE_HIP
include_aml_area_HEADERS += aml/area/hip.h
include_aml_dma_HEADERS += aml/dma/hip.h
endif

if HAVE_OPENCL
include_aml_area_HEADERS += aml/area/opencl.h
endif

if HAVE_ZE
include_aml_area_HEADERS += aml/area/ze.h
include_aml_dma_HEADERS += aml/dma/ze.h
include_aml_backend_HEADERS += aml/utils/backend/ze.h
endif

noinst_include_internaldir = $(includedir)/internal
noinst_include_internal_HEADERS = \
				  internal/utarray.h \
				  internal/uthash.h \
				  internal/utlist.h \
				  internal/utstack.h
