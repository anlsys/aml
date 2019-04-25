ACLOCAL_AMFLAGS = -I m4
SUBDIRS = src include tests

if ADD_BENCHMARKS
SUBDIRS += benchmarks
endif

if AML_DOXYGEN
SUBDIRS += doc
endif

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = aml.pc

EXTRA_DIST = autogen.sh aml.pc README.markdown

