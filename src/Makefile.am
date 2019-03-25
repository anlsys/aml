lib_LTLIBRARIES = libaml.la

libaml_la_SOURCES = aml.c

libaml_la_CFLAGS = -I$(top_srcdir)/include

libaml_la_LDFLAGS=

libaml_la_LIBADD = $(top_srcdir)/jemalloc/lib/libjemalloc-aml_pic.a

SUBDIRS=utils
libaml_la_LIBADD+=$(top_srcdir)/src/utils/libamlutils.la

SUBDIRS+=area
libaml_la_LIBADD+=$(top_srcdir)/src/area/libamlarea.la

SUBDIRS+=dma
libaml_la_LIBADD+=$(top_srcdir)/src/dma/libamldma.la

SUBDIRS+=tiling
libaml_la_LIBADD+=$(top_srcdir)/src/tiling/libamltiling.la

SUBDIRS+=scratch
libaml_la_LIBADD+=$(top_srcdir)/src/scratch/libamlscratch.la

SUBDIRS+=arena
libaml_la_LIBADD+=$(top_srcdir)/src/arena/libamlarena.la

