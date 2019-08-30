SUFFIXES=.c .cu
AM_CPPFLAGS = -I$(top_srcdir)/include $(PTHREAD_CFLAGS)
AM_LDFLAGS = $(PTHREAD_LIBS)
noinst_LTLIBRARIES=

#############################################
# .C sources

AREA_SOURCES = \
	       area/area.c \
	       area/linux.c

LAYOUT_SOURCES = \
	       layout/layout.c \
	       layout/dense.c \
	       layout/pad.c \
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
	      $(CUDA_AREA_SOURCES) \
	      aml.c

lib_LTLIBRARIES = libaml.la

libaml_la_LDFLAGS=
libaml_la_SOURCES=$(LIB_SOURCES)

#############################################
# Cuda sources

if BUILD_CUDA

# Build .c sources using cuda runtime library.
libaml_la_SOURCES+=area/cuda.c
libaml_la_LDFLAGS+=-lcudart

# Build .cu sources containing device code.
#
# CUDA_LO_FILES= # .lo files result of .cu files.
# CUDA_FLAGS=--x=cu
#
# .cu.lo: .cu
# 	$(NVCC) $(CUDA_FLAGS) -I$(top_srcdir)/include -dc -o $<.cu.o $<
# 	$(NVCC) $(CUDA_FLAGS) -I$(top_srcdir)/include -c -o $<.o $<
# 	$(top_builddir)/o2lo $@ $<.cu.o $<.o
#
# libamlcuda.la: $(CUDA_LO_FILES)
# 	$(LIBTOOL) --tag=CC --mode=link $(NVCC) -dlink -o $@ $< -lcudart
#
# libaml_la_LIBADD+=libamlcuda.la

endif
