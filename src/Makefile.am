AM_CPPFLAGS = -I$(top_srcdir)/include $(PTHREAD_CFLAGS)
AM_LDFLAGS = $(PTHREAD_LIBS)


AREA_SOURCES = \
	       area/area.c \
	       area/linux.c

LAYOUT_SOURCES = \
	       layout/layout.c \
	       layout/dense.c \
	       layout/reshape.c

DMA_SOURCES = \
	      dma/dma.c \
	      dma/dma_linux_par.c \
	      dma/dma_linux_seq.c

SCRATCH_SOURCES = \
		  scratch/scratch.c \
		  scratch/scratch_par.c \
		  scratch/scratch_seq.c

TILING_SOURCES = \
		 tiling/tiling.c \
		 tiling/tiling_1d.c \
		 tiling/tiling_2d.c

UTILS_SOURCES = \
		utils/bitmap.c \
		utils/error.c \
		utils/vector.c

LIB_SOURCES = \
	      $(AREA_SOURCES) \
	      $(DMA_SOURCES) \
	      $(SCRATCH_SOURCES) \
	      $(TILING_SOURCES) \
	      $(LAYOUT_SOURCES) \
	      $(UTILS_SOURCES) \
	      aml.c

lib_LTLIBRARIES = libaml.la
libaml_la_SOURCES = $(LIB_SOURCES)
