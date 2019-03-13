ACLOCAL_AMFLAGS = -I m4
SUBDIRS = src include tests

if ADD_BENCHMARKS
SUBDIRS += benchmarks
endif

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = aml.pc

install-recursive:
	$(MAKE) -C src install
	$(MAKE) install-am

EXTRA_DIST = autogen.sh aml.pc README.markdown

