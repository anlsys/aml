AM_CFLAGS = -I$(top_srcdir)/include -I. $(OPENMP_CFLAGS)
AM_LDFLAGS = ../src/libaml.la $(OPENMP_CFLAGS)

noinst_LIBRARIES = libutils.a
libutils_a_SOURCES = utils.c utils.h
LDADD = libutils.a

noinst_PROGRAMS = stream_add_pth_st \
		  stream_add_omp_st \
		  stream_add_omp_mt \
		  dgemm_vanilla \
		  dgemm_mkl \
		  dgemm_prefetch \
		  dgemm_noprefetch
