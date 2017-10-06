AM_CPPFLAGS = -I$(top_srcdir)/jemalloc/include
lib_LTLIBRARIES = libaml.la

LIBCSOURCES = aml.c area.c dma.c arena.c
LIBHSOURCES = aml.h

libaml_la_SOURCES = $(LIBCSOURCES) $(LIBHSOURCES)
libaml_la_LIBADD = $(top_srcdir)/jemalloc/lib/libjemalloc_pic.a
include_HEADERS = $(LIBHSOURCES)
