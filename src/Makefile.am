SUFFIXES=.c .cu
AM_CPPFLAGS = -I$(top_srcdir)/include -I$(top_srcdir)/excit/src $(PTHREAD_CFLAGS) $(OPENMP_CFLAGS)
AM_LDFLAGS = $(PTHREAD_LIBS) $(OPENMP_CFLAGS)
noinst_LTLIBRARIES=

#############################################
# valgrind support
@VALGRIND_CHECK_RULES@

#############################################
# .C sources

AREA_SOURCES = \
	       area/area.c \
	       area/linux.c

LAYOUT_SOURCES = \
	       layout/layout.c \
	       layout/dense.c \
	       layout/pad.c \
	       layout/reshape.c

DMA_SOURCES = \
	      dma/dma.c \
	      dma/linux.c \
	      dma/dma_linux_par.c \
	      dma/dma_linux_seq.c

TILING_SOURCES = \
		 tiling/tiling.c \
		 tiling/tiling_pad.c \
		 tiling/tiling_resize.c

UTILS_SOURCES = \
		utils/bitmap.c \
		utils/error.c \
		utils/vector.c \
		utils/queue.c \
		utils/async.c \
		utils/features.c

BACKEND_SOURCES = backend/linux.c

HIGHER_SOURCES = \
		replicaset/replicaset.c \
		higher/mapper.c

LIB_SOURCES = \
	      $(AREA_SOURCES) \
	      $(DMA_SOURCES) \
	      $(TILING_SOURCES) \
	      $(LAYOUT_SOURCES) \
	      $(UTILS_SOURCES) \
	      $(BACKEND_SOURCES) \
	      $(HIGHER_SOURCES) \
	      aml.c

lib_LTLIBRARIES = libaml.la

libaml_la_LDFLAGS=
libaml_la_SOURCES=$(LIB_SOURCES)

#############################################
# opencl sources

if HAVE_OPENCL
AM_CPPFLAGS += $(OPENCL_CFLAGS)
AM_LDFLAGS += $(OPENCL_LIBS)
libaml_la_SOURCES+=area/opencl.c
endif

#############################################
# hwloc sources

if HAVE_HWLOC
AM_CPPFLAGS += $(HWLOC_CFLAGS)
AM_LDFLAGS += $(HWLOC_LIBS)
libaml_la_SOURCES+=area/hwloc.c
libaml_la_SOURCES+=replicaset/hwloc.c
libaml_la_SOURCES+=backend/hwloc.c
endif

#############################################
# Cuda sources

if HAVE_CUDA
libcuda_la_SOURCES=area/cuda.c dma/cuda.c
noinst_LTLIBRARIES+=libcuda.la
libcuda_la_CPPFLAGS=$(AM_CPPFLAGS) $(CUDA_CFLAGS)
libcuda_la_LDFLAGS=$(AM_LDFLAGS) $(CUDA_LIBS)
libaml_la_LIBADD=libcuda.la
endif

#############################################
# Level Zero sources

if HAVE_ZE
AM_LDFLAGS += $(ZE_LIBS)
AM_CPPFLAGS += $(ZE_CFLAGS)
libaml_la_SOURCES+=area/ze.c
libaml_la_SOURCES+=dma/ze.c
libaml_la_SOURCES+=backend/ze.c
endif
