AM_COLOR_TESTS = yes

# add OpenMP to flags for test programs
AM_CFLAGS = -fopenmp -I$(top_srcdir)/src
AM_LDFLAGS = -fopenmp ../src/libaml.la

# valgrind support
if TEST_VALGRIND
TESTS_ENVIRONMENT= @LIBTOOL@ --mode=execute @VALGRIND@ --tool=memcheck -q --leak-check=full
endif

# all check programs
TST_PROGS = stream_add_pth stream_add_omp stream_vanilla

check_PROGRAMS = $(TST_PROGS)
TESTS = $(TST_PROGS)
