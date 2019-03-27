AM_CFLAGS=-I$(top_srcdir)/include

AM_CPPFLAGS=$(AM_CFLAGS)

noinst_LTLIBRARIES=libamldma.la

libamldma_la_SOURCES=\
               dma.c \
	       dma_linux_par.c \
	       dma_linux_seq.c

libamldma_la_CFLAGS=$(AM_CFLAGS)
libamldma_la_LDFLAGS=$(AM_CFLAGS)

