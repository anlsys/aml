AM_COLOR_TESTS = yes

# add OpenMP to flags for test programs
AM_CFLAGS = -fopenmp -I$(top_srcdir)/src
AM_LDFLAGS = -fopenmp ../src/libaml.la ../jemalloc/lib/libjemalloc-aml.so

# valgrind support
if TEST_VALGRIND
TESTS_ENVIRONMENT= @LIBTOOL@ --mode=execute @VALGRIND@ --tool=memcheck -q --leak-check=full
endif

UTILS_TESTS = vector

ARENA_JEMALLOC_TESTS = arena_jemalloc

AREA_LINUX_TESTS = linux_mbind \
		   linux_mmap \
		   linux_manager \
		   area_linux

AREA_POSIX_TESTS = area_posix

TILING_TESTS = tiling

BINDING_TESTS = binding

DMA_LINUX_TESTS = dma_linux_seq \
		  dma_linux_par

SCRATCH_TESTS = scratch_seq \
		scratch_par

# unit tests
UNIT_TESTS = $(ARENA_JEMALLOC_TESTS) \
	     $(UTILS_TESTS) \
	     $(TILING_TESTS) \
	     $(BINDING_TESTS) \
	     $(AREA_LINUX_TESTS) \
	     $(AREA_POSIX_TESTS) \
	     $(DMA_LINUX_TESTS) \
	     $(SCRATCH_TESTS)

# fonctional tests
FUNC_TESTS = stream_add_pth_st stream_add_omp_st

# all tests
TST_PROGS = $(UNIT_TESTS) $(FUNC_TESTS)
check_PROGRAMS = $(TST_PROGS)
TESTS = $(TST_PROGS)
