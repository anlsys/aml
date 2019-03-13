AM_CFLAGS=-I$(top_srcdir)/include
AM_CPPFLAGS=$(AM_CFLAGS)

check_PROGRAMS=
TESTS=

####################################################################
# Utils
####################################################################

test_bitmap_SOURCES=utils/test_bitmap.c
test_bitmap_CFLAGS=$(AM_CFLAGS)
test_bitmap_LDFLAGS=$(top_srcdir)/src/utils/libamlutils.la
check_PROGRAMS+=test_bitmap
TESTS+=test_bitmap

test_vector_SOURCES=utils/test_vector.c
test_vector_CFLAGS=$(AM_CFLAGS)
test_vector_LDFLAGS=$(top_srcdir)/src/utils/libamlutils.la
check_PROGRAMS+=test_vector
TESTS+=test_vector

####################################################################
# Tiling
####################################################################

test_tiling_SOURCES=tiling/test_tiling.c
test_tiling_CFLAGS=$(AM_CFLAGS)
test_tiling_LDFLAGS=$(top_srcdir)/src/libaml.la
check_PROGRAMS+=test_tiling
TESTS+=test_tiling

test_tiling_2d_SOURCES=tiling/test_tiling_2d.c
test_tiling_2d_CFLAGS=$(AM_CFLAGS)
test_tiling_2d_LDFLAGS=$(top_srcdir)/src/libaml.la
check_PROGRAMS+=test_tiling_2d
TESTS+=test_tiling_2d

####################################################################
# Scratch
####################################################################

test_scratch_seq_SOURCES=scratch/test_scratch_seq.c
test_scratch_seq_CFLAGS=$(AM_CFLAGS)
test_scratch_seq_LDFLAGS=$(top_srcdir)/src/libaml.la
check_PROGRAMS+=test_scratch_seq
TESTS+=test_scratch_seq

test_scratch_par_SOURCES=scratch/test_scratch_par.c
test_scratch_par_CFLAGS=$(AM_CFLAGS)
test_scratch_par_LDFLAGS=$(top_srcdir)/src/libaml.la
check_PROGRAMS+=test_scratch_par
TESTS+=test_scratch_par

####################################################################
# Dma
####################################################################

test_dma_linux_seq_SOURCES=dma/test_dma_linux_seq.c
test_dma_linux_seq_CFLAGS=$(AM_CFLAGS)
test_dma_linux_seq_LDFLAGS=$(top_srcdir)/src/libaml.la
check_PROGRAMS+=test_dma_linux_seq
TESTS+=test_dma_linux_seq

test_dma_linux_par_SOURCES=dma/test_dma_linux_par.c
test_dma_linux_par_CFLAGS=$(AM_CFLAGS)
test_dma_linux_par_LDFLAGS=$(top_srcdir)/src/libaml.la
check_PROGRAMS+=test_dma_linux_par
TESTS+=test_dma_linux_par

####################################################################
# Binding
####################################################################

test_binding_SOURCES=binding/test_binding.c
test_binding_CFLAGS=$(AM_CFLAGS)
test_binding_LDFLAGS=$(top_srcdir)/src/libaml.la
check_PROGRAMS+=test_binding
TESTS+=test_binding

####################################################################
# Area
####################################################################

test_linux_SOURCES=area/test_linux.c
test_linux_CFLAGS=$(AM_CFLAGS)
test_linux_LDFLAGS=$(top_srcdir)/src/libaml.la
check_PROGRAMS+=test_linux
TESTS+=test_linux

test_posix_SOURCES=area/test_posix.c
test_posix_CFLAGS=$(AM_CFLAGS)
test_posix_LDFLAGS=$(top_srcdir)/src/libaml.la
check_PROGRAMS+=test_posix
TESTS+=test_posix

test_mmap_SOURCES=area/test_mmap.c
test_mmap_CFLAGS=$(AM_CFLAGS)
test_mmap_LDFLAGS=$(top_srcdir)/src/libaml.la
check_PROGRAMS+=test_mmap
TESTS+=test_mmap

# test_mbind_SOURCES=area/test_mbind.c
# test_mbind_CFLAGS=$(AM_CFLAGS)
# test_mbind_LDFLAGS=$(top_srcdir)/src/libaml.la
# check_PROGRAMS+=test_mbind
# TESTS+=test_mbind

test_manager_SOURCES=area/test_manager.c
test_manager_CFLAGS=$(AM_CFLAGS)
test_manager_LDFLAGS=$(top_srcdir)/src/libaml.la
check_PROGRAMS+=test_manager
TESTS+=test_manager

