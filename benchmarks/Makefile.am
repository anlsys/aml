AM_CFLAGS = -I$(top_srcdir)/src -I. @OPENMP_FLAG@
AM_LDFLAGS = ../src/libaml.la ../jemalloc/lib/libjemalloc-aml.so @OPENMP_FLAG@

noinst_LIBRARIES = libutils.a
libutils_a_SOURCES = utils.c utils.h
LDADD = libutils.a

noinst_PROGRAMS = stream_add_pth_st \
		  stream_add_omp_st \
		  stream_add_omp_mt \
		  dgemm_mkl \
		  dgemm_prefetch \
		  dgemm_noprefetch
