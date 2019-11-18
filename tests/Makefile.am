AM_COLOR_TESTS = yes

AM_CFLAGS = -I$(top_srcdir)/include $(PTHREAD_CFLAGS)
AM_LDFLAGS = ../src/libaml.la $(PTHREAD_LIBS)
if HAVE_CUDA
# LIBS is used instead of AM_LDFLAGS on purpose
# AM_LDFLAGS appends flags before libraries added before LDADD.
# Thus, when linking with libaml.la, linking with cuda is not done.
LIBS += $(CUDA_CFLAGS)
LIBS += $(CUDA_LIBS)
endif

if HAVE_CUDA
# LIBS is used instead of AM_LDFLAGS on purpose
# AM_LDFLAGS appends flags before libraries added before LDADD.
# Thus, when linking with libaml.la, linking with cuda is not done.
LIBS += $(CUDA_CFLAGS)
LIBS += $(CUDA_LIBS)
AM_CFLAGS += $(CUDA_CFLAGS)
AM_LDFLAGS += $(CUDA_LIBS)
endif

# valgrind support
if TEST_VALGRIND
TESTS_ENVIRONMENT= @LIBTOOL@ --mode=execute @VALGRIND@ --tool=memcheck -q --leak-check=full
endif

UTILS_TESTS = \
	      utils/test_bitmap \
	      utils/test_inner_malloc \
	      utils/test_vector \
	      utils/test_version

AREA_TESTS = \
	area/test_area \
	area/test_linux

noinst_LTLIBRARIES = liblayout_test.la
liblayout_test_la_CPPFLAGS =
liblayout_test_la_LDFLAGS =
liblayout_test_la_SOURCES = \
	layout/coords.c \
	layout/dense.c \
	layout/layout.c \
	layout/reshape.c
liblayout_test_la_LIBADD = ../src/libaml.la
LDADD = liblayout_test.la

LAYOUT_TESTS = \
	layout/test_coords \
	layout/test_dense \
	layout/test_reshape \
	layout/test_pad

TILING_TESTS = tiling/test_tiling

DMA_LINUX_TESTS = dma/test_dma_linux_seq \
		  dma/test_dma_linux_par

SCRATCH_TESTS = scratch/test_scratch_seq \
		scratch/test_scratch_par

if HAVE_CUDA
AREA_TESTS += area/test_cuda
endif

# unit tests
UNIT_TESTS = $(UTILS_TESTS) \
	     $(TILING_TESTS) \
	     $(LAYOUT_TESTS) \
	     $(BINDING_TESTS) \
	     $(AREA_TESTS) \
	     $(DMA_LINUX_TESTS) \
	     $(SCRATCH_TESTS)

# all tests
TST_PROGS = $(UNIT_TESTS)
check_PROGRAMS = $(TST_PROGS)
TESTS = $(TST_PROGS)
