AM_COLOR_TESTS = yes

AM_CFLAGS = -I$(top_srcdir)/include -I$(top_builddir)/include -I$(top_srcdir)/excit/src $(PTHREAD_CFLAGS) $(OPENMP_CFLAGS)
AM_LDFLAGS = ../src/libaml.la $(top_builddir)/excit/src/libexcit.la $(PTHREAD_LIBS) $(OPENMP_CFLAGS)

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
AM_CFLAGS += $(HWLOC_CFLAGS)
AM_LDFLAGS += $(HWLOC_LIBS)
endif

if HAVE_ZE
AM_LDFLAGS += $(ZE_LIBS)
AM_CFLAGS += $(ZE_CFLAGS)
endif

# valgrind support
@VALGRIND_CHECK_RULES@

noinst_LIBRARIES = libkernel.a

UTILS=utils.c

BLAS_UTILS=\
	       blas/l1_kernel.h \
	       blas/l1_kernel.c \
	       blas/verify_l1.h \
	       blas/verify_l1.c

ALLOCATOR_UTILS=\
	       ../tests/allocator/dummy_area.h \
	       ../tests/allocator/dummy_area.c \
	       ../tests/allocator/test_allocator.h \
	       ../tests/allocator/test_allocator.c \
	       allocator/allocator.h \
	       allocator/allocator.c

BLAS_BENCHMARKS=\
	       blas/flat_l1\
	       blas/tiled_l1

ALLOCATOR_BENCHMARKS=\
	       allocator/benchmark_area \
	       allocator/benchmark_buddy


libkernel_a_SOURCES=$(UTILS) \
	       $(BLAS_UTILS) \
	       $(ALLOCATOR_UTILS)

LDADD = libkernel.a

BENCHMARKS = \
	       $(BLAS_BENCHMARKS) \
	       $(ALLOCATOR_BENCHMARKS)

# all tests
check_PROGRAMS = $(BENCHMARKS)
TESTS = $(BENCHMARKS)

# phony target to allow us to compile the check programs without running the
# tests. Required for any environment where we are cross-compiling
check-programs-local: $(BENCHMARKS)
