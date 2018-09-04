AM_CFLAGS = -I$(top_srcdir)/src -I. @OPENMP_FLAG@
AM_LDFLAGS = ../src/libaml.la ../jemalloc/lib/libjemalloc-aml.so @OPENMP_FLAG@

noinst_LIBRARIES = libutils.a
libutils_a_SOURCES = utils.c utils.h
LDADD = libutils.a

noinst_PROGRAMS = stream_add_pth_st \
		  stream_add_omp_st \
		  stream_add_omp_mt \
		  dgemm_vanilla \
		  dgemm_mkl \
		  dgemm_cblas \
		  dgemm_prefetch \
		  dgemm_prefetch_blocked \
		  dgemm_noprefetch \
		  dgemm_noprefetch_task \
		  dgemm_noprefetch_gemm256_task \
		  dgemm_noprefetch_gemm320_task \
		  dgemm_noprefetch_gemm384_task \
		  dgemm_noprefetch_gemm448_task \
		  dgemm_noprefetch_gemm512_task \
		  dgemm_noprefetch_gemm256_cblas_task \
		  dgemm_noprefetch_gemm320_cblas_task \
		  dgemm_noprefetch_gemm384_cblas_task \
		  dgemm_noprefetch_gemm448_cblas_task \
		  dgemm_noprefetch_gemm512_cblas_task
#		  dgemm_noprefetch_gemm128_task
#		  dgemm_noprefetch_gemm128

gem_dir = ./gemm

#dgemm_noprefetch_gemm128_task_LDADD = gemm_128_128_12820180820_58220_1hgw8bn.o
dgemm_noprefetch_gemm256_task_LDADD = $(gem_dir)/gemm_256_256_256.o
dgemm_noprefetch_gemm320_task_LDADD = $(gem_dir)/gemm_320_320_320.o
dgemm_noprefetch_gemm384_task_LDADD = $(gem_dir)/gemm_384_384_384.o
dgemm_noprefetch_gemm448_task_LDADD = $(gem_dir)/gemm_448_448_448.o
dgemm_noprefetch_gemm512_task_LDADD = $(gem_dir)/gemm_512_512_512.o
dgemm_cblas_LDFLAGS = $(AM_LDFLAGS) -lblas
dgemm_noprefetch_gemm256_cblas_task_LDFLAGS = $(AM_LDFLAGS) -lblas
dgemm_noprefetch_gemm320_cblas_task_LDFLAGS = $(AM_LDFLAGS) -lblas
dgemm_noprefetch_gemm384_cblas_task_LDFLAGS = $(AM_LDFLAGS) -lblas
dgemm_noprefetch_gemm448_cblas_task_LDFLAGS = $(AM_LDFLAGS) -lblas
dgemm_noprefetch_gemm512_cblas_task_LDFLAGS = $(AM_LDFLAGS) -lblas
dgemm_prefetch_blocked_LDADD = $(gem_dir)/gemm_256_256_256.o $(gem_dir)/gemm_320_320_320.o $(gem_dir)/gemm_384_384_384.o $(gem_dir)/gemm_448_448_448.o $(gem_dir)/gemm_512_512_512.o
dgemm_vanilla_LDFLAGS = $(AM_LDFLAGS) -lmkl_rt
dgemm_mkl_LDFLAGS = $(AM_LDFLAGS) -lmkl_rt
dgemm_noprefetch_LDFLAGS = $(AM_LDFLAGS) -lmkl_rt
dgemm_noprefetch_task_LDFLAGS = $(AM_LDFLAGS) -lmkl_rt
dgemm_prefetch_LDFLAGS = $(AM_LDFLAGS) -lmkl_rt
#dgemm_noprefetch_gemm128_LDADD = gemm_128_128_12820180820_58220_1hgw8bn.o
