AM_CFLAGS=-I$(top_srcdir)/jemalloc/include -I$(top_srcdir)/include

AM_CPPFLAGS=$(AM_CFLAGS)

noinst_LTLIBRARIES=libamlarena.la

libamlarena_la_SOURCES=\
arena.c\
arena_jemalloc.c

libamlarena_la_CFLAGS=$(AM_CFLAGS)

