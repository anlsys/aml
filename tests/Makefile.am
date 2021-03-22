
AM_COLOR_TESTS = yes

AM_CFLAGS = -I$(top_srcdir)/include -I$(top_srcdir)/excit/src $(PTHREAD_CFLAGS) $(OPENMP_CFLAGS)
AM_LDFLAGS = ../src/libaml.la $(top_builddir)/excit/src/libexcit.la $(PTHREAD_LIBS) $(OPENMP_CFLAGS)

if HAVE_OPENCL
LIBS += $(OpenCL_LIBS)
AM_CFLAGS += $(OpenCL_CFLAGS)
AM_LDFLAGS += $(OpenCL_LIBS)
endif

if HAVE_HWLOC
LIBS += $(HWLOC_LIBS)
AM_CFLAGS += $(HWLOC_CFLAGS)
AM_LDFLAGS += $(HWLOC_LIBS)
endif

if HAVE_ZE
LIBS += $(libze_loader_LIBS)
AM_LDFLAGS += $(libze_loader_LIBS)
AM_CFLAGS += $(libze_loader_CFLAGS)
endif

# valgrind support
@VALGRIND_CHECK_RULES@

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

libreplicaset_test_la_SOURCES = \
	replicaset/test_replicaset.h \
	replicaset/common.c

LDADD = liblayout_test.la libreplicaset_test.la

LAYOUT_TESTS = \
	layout/test_coords \
	layout/test_dense \
	layout/test_reshape \
	layout/test_pad

TILING_TESTS = tiling/test_tiling

DMA_TESTS = dma/test_dma_linux_seq \
		  dma/test_dma_linux_par


REPLICASET_TESTS =

if HAVE_OPENCL
AREA_TESTS += area/test_opencl
endif

if HAVE_ZE
AREA_TESTS += area/test_ze
endif

if HAVE_HWLOC
AREA_TESTS += area/test_hwloc
REPLICASET_TESTS += replicaset/test_hwloc
endif

# unit tests
UNIT_TESTS = $(UTILS_TESTS) \
	     $(TILING_TESTS) \
	     $(LAYOUT_TESTS) \
	     $(BINDING_TESTS) \
	     $(AREA_TESTS) \
	     $(REPLICASET_TESTS) \
	     $(DMA_TESTS)

# all tests
TST_PROGS = $(UNIT_TESTS)

if HAVE_CUDA
test_area_cuda_SOURCES=area/test_cuda.c
test_area_cuda_CPPFLAGS= $(AM_CFLAGS) $(CUDA_CFLAGS)
test_area_cuda_LDFLAGS=$(AM_LDFLAGS) $(CUDA_LIBS)

test_dma_cuda_SOURCES=dma/test_cuda.c
test_dma_cuda_CPPFLAGS= $(AM_CFLAGS) $(CUDA_CFLAGS)
test_dma_cuda_LDFLAGS=$(AM_LDFLAGS) $(CUDA_LIBS)
TST_PROGS+=test_dma_cuda test_area_cuda
endif

check_PROGRAMS = $(TST_PROGS)
check_LTLIBRARIES = liblayout_test.la libreplicaset_test.la
TESTS = $(TST_PROGS)

# phony target to allow us to compile the check programs without running the
# tests. Required for any environment where we are cross-compiling
check-programs-local: $(TST_PROGS)
