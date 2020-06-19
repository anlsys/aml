SUFFIXES=.c .cu
AM_CPPFLAGS = -I$(top_srcdir)/include $(PTHREAD_CFLAGS)
AM_LDFLAGS = $(PTHREAD_LIBS)
noinst_LTLIBRARIES=

#############################################
# valgrind support
@VALGRIND_CHECK_RULES@

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
		 tiling/tiling_pad.c \
		 tiling/tiling_resize.c

UTILS_SOURCES = \
		utils/bitmap.c \
		utils/error.c \
		utils/vector.c \
		utils/queue.c \
		utils/async.c \
		utils/features.c

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

if HAVE_CUDA

AM_CPPFLAGS += $(CUDA_CFLAGS)
AM_LDFLAGS += $(CUDA_LIBS)

# Build .c sources using cuda runtime library.
libaml_la_SOURCES+=area/cuda.c

# Build files not requiring additional libraries
LAYOUT_SOURCES+=layout/cuda.c

# Build .cu sources containing device code.
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

#############################################
# hwloc sources

if HAVE_HWLOC
AM_CPPFLAGS += $(HWLOC_CFLAGS)
AM_LDFLAGS += $(HWLOC_LIBS)

libaml_la_SOURCES+=area/hwloc.c
endif
