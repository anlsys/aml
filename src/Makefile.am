AM_CPPFLAGS = -I$(top_srcdir)/jemalloc/include @EXCIT_CFLAGS@
lib_LTLIBRARIES = libaml.la

ARENA_JEMALLOC_CSOURCES = arena_jemalloc.c

AREA_LINUX_CSOURCES = area_linux.c \
		      area_linux_manager.c \
		      area_linux_mbind.c \
		      area_linux_mmap.c

AREA_POSIX_CSOURCES = area_posix.c

LAYOUT_CSOURCES = layout.c \
		  layout_dense.c \
		  layout_pad.c \
		  layout_reshape.c

TILING_CSOURCES = tiling.c \
		  tiling_1d.c \
		  tiling_2d.c

BINDING_CSOURCES = binding.c \
		   binding_single.c \
		   binding_interleave.c

DMA_CSOURCES = dma.c \
	       dma_linux_par.c \
	       dma_linux_seq.c

SCRATCH_CSOURCES = scratch.c \
		   scratch_seq.c \
		   scratch_par.c

UTILS_CSOURCES = vector.c

LIBCSOURCES = aml.c area.c arena.c \
	      $(UTILS_CSOURCES) \
	      $(ARENA_JEMALLOC_CSOURCES) \
	      $(AREA_LINUX_CSOURCES) \
	      $(AREA_POSIX_CSOURCES) \
	      $(TILING_CSOURCES) \
	      $(BINDING_CSOURCES) \
	      $(DMA_CSOURCES) \
	      $(SCRATCH_CSOURCES) \
	      $(LAYOUT_CSOURCES) \
	      copy.c

LIBHSOURCES = aml.h aml-layout.h aml-layout-dense.h aml-layout-pad.h aml-layout-reshape.h aml-copy.h

libaml_la_SOURCES = $(LIBCSOURCES) $(LIBHSOURCES)
libaml_la_LIBADD = -L$(top_srcdir)/jemalloc/lib/ -ljemalloc-aml @EXCIT_LIBS@
include_HEADERS = $(LIBHSOURCES)
