AM_CPPFLAGS = \
	      -I$(top_srcdir)/include  \
	      -I$(top_builddir)/include \
	      -I$(top_srcdir)/excit/src \
	      $(PTHREAD_CFLAGS) $(OPENMP_CFLAGS)
AM_LDFLAGS = $(PTHREAD_LIBS) $(OPENMP_LIBS)
CC=@MPICC@

lib_LTLIBRARIES = libaml-mpi.la

libaml_mpi_la_SOURCES = area.c

libaml_mpi_la_LIBADD = $(top_builddir)/src/libaml.la
