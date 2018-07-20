ACLOCAL_AMFLAGS = -I m4
SUBDIRS = jemalloc src tests

if ADD_BENCHMARKS
SUBDIRS += benchmarks
endif

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = aml.pc

# remove jemalloc tests from make check
check-recursive:
	$(MAKE) -C tests check

install-recursive:
	$(MAKE) -C jemalloc install_lib
	$(MAKE) -C src install
	$(MAKE) install-am

EXTRA_DIST = autogen.sh aml.pc README.markdown
