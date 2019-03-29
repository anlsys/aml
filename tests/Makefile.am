AM_COLOR_TESTS = yes

# add OpenMP to flags for test programs
AM_CFLAGS = -I$(top_srcdir)/include
AM_LDFLAGS = ../src/libaml.la

# valgrind support
if TEST_VALGRIND
TESTS_ENVIRONMENT= @LIBTOOL@ --mode=execute @VALGRIND@ --tool=memcheck -q --leak-check=full
endif

UTILS_TESTS = utils/test_vector \
	      utils/test_bitmap \
	      utils/test_version

AREA_TESTS = \
	area/test_area \
	area/test_linux

TILING_TESTS = tiling/test_tiling \
	       tiling/test_tiling_2d

DMA_LINUX_TESTS = dma/test_dma_linux_seq \
		  dma/test_dma_linux_par

SCRATCH_TESTS = scratch/test_scratch_seq \
		scratch/test_scratch_par

# unit tests
UNIT_TESTS = $(UTILS_TESTS) \
	     $(TILING_TESTS) \
	     $(BINDING_TESTS) \
	     $(AREA_TESTS) \
	     $(DMA_LINUX_TESTS) \
	     $(SCRATCH_TESTS)

# all tests
TST_PROGS = $(UNIT_TESTS)
check_PROGRAMS = $(TST_PROGS)
TESTS = $(TST_PROGS)
