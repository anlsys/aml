AM_COLOR_TESTS = yes

AM_CFLAGS = -I$(top_srcdir)/include $(PTHREAD_CFLAGS) $(OPENMP_CFLAGS)
AM_LDFLAGS = ../src/libaml.la $(PTHREAD_LIBS) $(OPENMP_CFLAGS)

if HAVE_HWLOC
AM_CFLAGS += $(HWLOC_CFLAGS)
AM_LDFLAGS += $(HWLOC_LIBS)
endif

if HAVE_CUDA
LIBS += $(CUDA_CFLAGS)
LIBS += $(CUDA_LIBS)
AM_CFLAGS += $(CUDA_CFLAGS)
AM_LDFLAGS += $(CUDA_LIBS)
endif

noinst_LIBRARIES = libkernel.a
libkernel_a_SOURCES = utils.c utils.h blas_l1_kernel.c blas_l1_kernel.h verify_blas_l1.c verify_blas_l1.h
LDADD = libkernel.a

NOPREFETCH = noprefetch/flat_blas_l1 \
	     noprefetch/tiled_blas_l1

BENCHMARKS = $(NOPREFETCH)

# all tests
check_PROGRAMS = $(BENCHMARKS)
TESTS = $(BENCHMARKS)
