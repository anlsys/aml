AM_CFLAGS=-I$(top_srcdir)/include
AM_CPPFLAGS=$(AM_CFLAGS)
check_PROGRAMS=
TESTS=

####################################################################
# Utils
####################################################################

test_bitmap_SOURCES=utils/test_bitmap.c
test_bitmap_LDFLAGS=$(top_srcdir)/src/utils/libamlutils.la
if HAVE_HWLOC
test_bitmap_LDFLAGS+=-lhwloc
endif
check_PROGRAMS+=test_bitmap
TESTS+=test_bitmap

test_vector_SOURCES=utils/test_vector.c
test_vector_LDFLAGS=$(top_srcdir)/src/utils/libamlutils.la
check_PROGRAMS+=test_vector
TESTS+=test_vector

####################################################################
# Tiling
####################################################################

test_tiling_SOURCES=tiling/test_tiling.c
test_tiling_LDFLAGS=$(top_srcdir)/src/libaml.la
check_PROGRAMS+=test_tiling
TESTS+=test_tiling

test_tiling_2d_SOURCES=tiling/test_tiling_2d.c
test_tiling_2d_LDFLAGS=$(top_srcdir)/src/libaml.la
check_PROGRAMS+=test_tiling_2d
TESTS+=test_tiling_2d

####################################################################
# Scratch
####################################################################

test_scratch_seq_SOURCES=scratch/test_scratch_seq.c
test_scratch_seq_LDFLAGS=$(top_srcdir)/src/libaml.la
check_PROGRAMS+=test_scratch_seq
TESTS+=test_scratch_seq

test_scratch_par_SOURCES=scratch/test_scratch_par.c
test_scratch_par_LDFLAGS=$(top_srcdir)/src/libaml.la
check_PROGRAMS+=test_scratch_par
TESTS+=test_scratch_par

####################################################################
# Dma
####################################################################

test_dma_linux_seq_SOURCES=dma/test_dma_linux_seq.c
test_dma_linux_seq_LDFLAGS=$(top_srcdir)/src/libaml.la
check_PROGRAMS+=test_dma_linux_seq
TESTS+=test_dma_linux_seq

test_dma_linux_par_SOURCES=dma/test_dma_linux_par.c
test_dma_linux_par_LDFLAGS=$(top_srcdir)/src/libaml.la
check_PROGRAMS+=test_dma_linux_par
TESTS+=test_dma_linux_par

####################################################################
# Area
####################################################################


