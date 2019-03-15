AM_CFLAGS=-I$(top_srcdir)/include
AM_CPPFLAGS=$(AM_CFLAGS)

noinst_LTLIBRARIES=libamlscratch.la

libamlscratch_la_SOURCES=\
scratch.c \
scratch_seq.c \
scratch_par.c \
scratch_double.c \
aml-scratch-double.h

libamlscratch_la_CFLAGS=$(AM_CFLAGS) $(PTHREAD_CFLAGS)
libamlscratch_la_LDFLAGS=$(PTHREAD_LIBS)
