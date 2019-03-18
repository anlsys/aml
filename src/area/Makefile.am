AM_CFLAGS=-I$(top_srcdir)/include

AM_CPPFLAGS=$(AM_CFLAGS)

noinst_LTLIBRARIES=libamlarea.la

libamlarea_la_LDFLAGS=

libamlarea_la_CFLAGS=\
	$(AM_CFLAGS)\
	-Wno-incompatible-pointer-types\
	-Wno-discarded-qualifiers

libamlarea_la_SOURCES=\
	area.c \
	linux.c

if HAVE_LINUX_NUMA
libamlarea_la_LDFLAGS+=-lnuma
endif

if HAVE_HWLOC
libamlarea_la_SOURCES+=hwloc.c
libamlarea_la_LDFLAGS+=-lhwloc
endif

