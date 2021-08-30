ACLOCAL_AMFLAGS = -I m4
SUBDIRS = src include tests doc benchmarks excit

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = aml.pc

EXTRA_DIST = autogen.sh aml.pc README.markdown .version

BUILT_SOURCES = $(top_srcdir)/.version
$(top_srcdir)/.version:
	echo $(VERSION) > $@-t && mv $@-t $@
dist-hook:
	echo $(VERSION) > $(distdir)/.tarball-version
