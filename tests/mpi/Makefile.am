AM_COLOR_TESTS = yes
AM_CFLAGS = -I$(top_srcdir)/include -I$(top_builddir)/include -I$(top_srcdir)/excit/src $(PTHREAD_CFLAGS) $(OPENMP_CFLAGS)
AM_LDFLAGS = ../../src/libaml.la ../../src/mpi/libaml-mpi.la $(top_builddir)/excit/src/libexcit.la $(PTHREAD_LIBS) $(OPENMP_LIBS) $(MPILIBS)
CC=$(MPICC)

test_area_mpi_SOURCES=test_area.c
test_area_mpi_CPPFLAGS= $(AM_CFLAGS)

TST_PROGS = test_area_mpi

check_PROGRAMS = $(TST_PROGS)
TESTS = $(TST_PROGS)

# phony target to allow us to compile the check programs without running the
# tests. Required for any environment where we are cross-compiling
check-programs-local: $(TST_PROGS)
