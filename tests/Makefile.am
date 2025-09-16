
AM_COLOR_TESTS = yes

AM_CFLAGS = -I$(top_srcdir)/include -I$(top_builddir)/include -I$(top_srcdir)/excit/src $(PTHREAD_CFLAGS) $(OPENMP_CFLAGS)
AM_LDFLAGS = ../src/libaml.la $(top_builddir)/excit/src/libexcit.la $(PTHREAD_LIBS) $(OPENMP_LIBS)
SUBDIRS =

if HAVE_CUDA
LIBS += $(CUDA_LIBS)
AM_LDFLAGS += $(CUDA_LIBS)
AM_CFLAGS += $(CUDA_CFLAGS)
endif

if HAVE_OPENCL
LIBS += $(OPENCL_LIBS)
AM_CFLAGS += $(OPENCL_CFLAGS)
AM_LDFLAGS += $(OPENCL_LIBS)
endif

if HAVE_HWLOC
LIBS += $(HWLOC_LIBS)
AM_CFLAGS += $(HWLOC_CFLAGS)
AM_LDFLAGS += $(HWLOC_LIBS)
endif

if HAVE_ZE
LIBS += $(ZE_LIBS)
AM_LDFLAGS += $(ZE_LIBS)
AM_CFLAGS += $(ZE_CFLAGS)
endif

if HAVE_HIP
LIBS += $(HIP_LIBS)
AM_LDFLAGS += $(HIP_LIBS)
AM_CFLAGS += $(HIP_CFLAGS)
endif

# valgrind support
@VALGRIND_CHECK_RULES@
VALGRIND_SUPPRESSIONS_FILES=../.valgrind.supp

UTILS_TESTS = \
	      utils/test_bitmap \
	      utils/test_inner_malloc \
	      utils/test_vector \
	      utils/test_async \
	      utils/test_queue \
	      utils/test_version

AREA_TESTS = \
	area/test_area \
	area/test_linux

liblayout_test_la_SOURCES = \
	layout/test_layout.h \
	layout/coords.c \
	layout/dense.c \
	layout/layout.c \
	layout/reshape.c

liballocator_test_la_SOURCES = \
	allocator/dummy_area.c \
	allocator/test_allocator.c \
	../benchmarks/utils.c \
	../benchmarks/utils.h

libdma_test_la_SOURCES = \
	dma/test_dma.h \
	dma/test_dma.c

libreplicaset_test_la_SOURCES = \
	replicaset/test_replicaset.h \
	replicaset/common.c

LDADD = libdma_test.la \
	liblayout_test.la \
	libreplicaset_test.la \
	liballocator_test.la

LAYOUT_TESTS = \
	layout/test_coords \
	layout/test_dense \
	layout/test_reshape \
	layout/test_pad

TILING_TESTS = \
	tiling/test_tiling

DMA_TESTS = dma/test_linux \
	    dma/test_multiplex

HIGHER_TESTS = \
	higher/test_mapper \
	higher/test_mapper-visitor \
	higher/test_mapper-creator \
	higher/test_mapper-deepcopy \
	higher/test_mapper-replicaset \
	allocator/test_area \
	allocator/test_buddy \
	allocator/test_sized

if HAVE_CUDA
AREA_TESTS+=area/test_cuda
AREA_TESTS+=dma/test_cuda
endif

if HAVE_OPENCL
AREA_TESTS += area/test_opencl
endif

if HAVE_ZE
AREA_TESTS += area/test_ze
DMA_TESTS += dma/test_ze dma/test-aml-dma-ze
endif

if HAVE_HWLOC
AREA_TESTS += area/test_hwloc
HIGHER_TESTS += replicaset/test_hwloc
endif

if HAVE_HIP
AREA_TESTS+=area/test_hip
AREA_TESTS+=dma/test_hip
endif

# unit tests
UNIT_TESTS = $(UTILS_TESTS) \
	     $(TILING_TESTS) \
	     $(LAYOUT_TESTS) \
	     $(BINDING_TESTS) \
	     $(AREA_TESTS) \
	     $(HIGHER_TESTS) \
	     $(DMA_TESTS)

# all tests
TST_PROGS = $(UNIT_TESTS)

#TODO: odd, figure out if it's actually required
if HAVE_CUDA
test_area_cuda_SOURCES=area/test_cuda.c
test_area_cuda_CPPFLAGS= $(AM_CFLAGS) $(CUDA_CFLAGS)
test_area_cuda_LDFLAGS=$(AM_LDFLAGS) $(CUDA_LIBS)

test_dma_cuda_SOURCES=dma/test_cuda.c
test_dma_cuda_CPPFLAGS= $(AM_CFLAGS) $(CUDA_CFLAGS)
test_dma_cuda_LDFLAGS=$(AM_LDFLAGS) $(CUDA_LIBS)
TST_PROGS+=test_dma_cuda test_area_cuda
endif

if HAVE_MPI
SUBDIRS += mpi
endif


check_PROGRAMS = $(TST_PROGS)
check_LTLIBRARIES = \
	libdma_test.la \
	liblayout_test.la \
	libreplicaset_test.la \
	liballocator_test.la

TESTS = $(TST_PROGS)

# phony target to allow us to compile the check programs without running the
# tests. Required for any environment where we are cross-compiling
check-programs-local: $(TST_PROGS)
