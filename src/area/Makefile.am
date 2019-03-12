AM_CFLAGS=-I$(top_srcdir)/include

AM_CPPFLAGS=$(AM_CFLAGS)

noinst_LTLIBRARIES=libamlarea.la

libamlarea_la_SOURCES=\
	area.c \
        area_linux.c \
        area_linux_manager.c \
        area_linux_mbind.c \
        area_linux_mmap.c \
	area_posix.c

libamlarea_la_CFLAGS=$(AM_CFLAGS)

