ACLOCAL_AMFLAGS = -I m4
SUBDIRS = jemalloc src tests

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = aml.pc

# remove jemalloc tests from make check
check-recursive:
	$(MAKE) -C tests check


EXTRA_DIST = autogen.sh aml.pc README.markdown
