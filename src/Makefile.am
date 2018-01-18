AM_CPPFLAGS = -I$(top_srcdir)/jemalloc/include
lib_LTLIBRARIES = libaml.la

ARENA_JEMALLOC_CSOURCES = arena_jemalloc.c

AREA_LINUX_CSOURCES = area_linux.c \
		      area_linux_manager.c \
		      area_linux_mbind.c \
		      area_linux_mmap.c

AREA_POSIX_CSOURCES = area_posix.c

LIBCSOURCES = aml.c area.c arena.c \
	      $(ARENA_JEMALLOC_CSOURCES) \
	      $(AREA_LINUX_CSOURCES) \
	      $(AREA_POSIX_CSOURCES)
LIBHSOURCES = aml.h

libaml_la_SOURCES = $(LIBCSOURCES) $(LIBHSOURCES)
libaml_la_LIBADD = $(top_srcdir)/jemalloc/lib/libjemalloc_pic.a
include_HEADERS = $(LIBHSOURCES)
