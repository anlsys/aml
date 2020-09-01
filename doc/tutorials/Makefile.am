AM_COLOR_TESTS = yes

AM_CFLAGS = -I$(top_srcdir)/include -I$(top_srcdir)/excit/src $(PTHREAD_CFLAGS)  $(OPENMP_CFLAGS)
AM_LDFLAGS = $(top_builddir)/src/libaml.la $(top_builddir)/excit/src/libexcit.la $(PTHREAD_LIBS) $(OPENMP_CFLAGS)

if HAVE_CUDA
# LIBS is used instead of AM_LDFLAGS on purpose
# AM_LDFLAGS appends flags before libraries added before LDADD.
# Thus, when linking with libaml.la, linking with cuda is not done.
LIBS += $(CUDA_CFLAGS)
LIBS += $(CUDA_LIBS)
AM_CFLAGS += $(CUDA_CFLAGS)
AM_LDFLAGS += $(CUDA_LIBS)
endif

if HAVE_HWLOC
LIBS += $(HWLOC_LIBS)
AM_CFLAGS += $(HWLOC_CFLAGS)
AM_LDFLAGS += $(HWLOC_LIBS)
endif

# valgrind support
@VALGRIND_CHECK_RULES@

HELLO_TESTS = \
	      hello_world/0_hello

AREA_TESTS = \
	      area/1_custom_interleave_area

LAYOUT_TESTS = \
	       layouts/0_dense_layout \
	       layouts/1_layout_reshape \
	       layouts/2_aos_soa

DMA_TESTS = \
	      dma/0_example \
	      dma/1_reduction

TILING_TESTS = \
	       tiling/0_generic_resize_tiling \
	       tiling/1_dgemm

if HAVE_CUDA
AREA_TESTS += area/0_aml_area_cuda
endif

if HAVE_HWLOC
AREA_TESTS += area/2_aml_area_hwloc
endif

UNIT_TESTS = $(HELLO_TESTS) $(AREA_TESTS) $(LAYOUT_TESTS) $(DMA_TESTS) $(TILING_TESTS)

check_PROGRAMS = $(UNIT_TESTS)
TESTS = $(UNIT_TESTS)

# phony target to allow us to compile the check programs without running the
# tests. Required for any environment where we are cross-compiling
check-programs-local: $(UNIT_TESTS)
