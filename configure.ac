# see semver.org for version info
AC_INIT([aml],[0.0.1],[swann@anl.gov])

# are we in the right source dir ?
AC_CONFIG_SRCDIR([include/aml.h])

# build artefacts in separate dir
AC_CONFIG_AUX_DIR([m4])
AC_CONFIG_MACRO_DIR([m4])

# automake should fail on any error
AM_INIT_AUTOMAKE([-Wall -Werror foreign subdir-objects 1.12])
AM_PROG_AR

# check for libtool
LT_INIT

# check for programs
AC_LANG([C])
AC_PROG_CC
AC_PROG_CC_STDC
AM_PROG_CC_C_O
AC_PROG_CPP
AC_TYPE_SIZE_T
AC_TYPE_INTPTR_T

# support for testing with valgrind
AC_ARG_ENABLE(valgrind,
[AS_HELP_STRING([--enable-valgrind],[Also valgrind on checks (default is no).])],
[valgrind=true],[valgrind=false])

if [[ "x$valgrind" = xtrue ]]; then
	AC_PATH_PROG(VALGRIND, valgrind, no)
	if [[  "x$VALGRIND" = xno ]]; then
		AC_MSG_ERROR([Valgrind not found in PATH. ])
	fi
fi
AM_CONDITIONAL([TEST_VALGRIND],[test "x$valgrind" = xtrue])

# support for compiling benchmarks
AC_ARG_ENABLE(benchmarks,
[AS_HELP_STRING([--enable-benchmarks],[Compile additional benchmarks (default is no).])],
[benchmarks=true],[benchmarks=false])

if [[ "x$benchmarks" = xtrue ]]; then
	AC_OPENMP
	AC_CHECK_LIB(m, sqrt)
fi
AM_CONDITIONAL([ADD_BENCHMARKS],[test "x$benchmarks" = xtrue])


# NUMA support
AC_CHECK_HEADERS(numa.h)
AC_CHECK_HEADERS(numaif.h)
AC_CHECK_LIB(numa, move_pages)
AC_CHECK_LIB(dl, dlopen)

# add pthread support.
# doc in m4/ax_pthread.m4. Defines automake PTHREAD_CFLAGS and PTHREAD_LIBS
AX_PTHREAD
CC="$PTHREAD_CC"

# Hwloc dependency
AC_CHECK_HEADER([hwloc.h],\
                [hwloc_header_happy=yes; \
		 LIBS="$LIBS -lhwloc"],\
	        [hwloc_header_happy=no])
		
AC_CHECK_LIB([hwloc],\
             [hwloc_get_api_version],\
	     [hwloc_lib_happy=yes],\
	     [hwloc_lib_happy=no])

AC_RUN_IFELSE(
       [AC_LANG_PROGRAM([#include <hwloc.h>],\
[#if HWLOC_API_VERSION < 0x00020000
return -1;
#else
if(hwloc_get_api_version() < 0x20000)
    return -1;
#endif])],\
       [hwloc_version_happy=yes],\
       [hwloc_version_happy=no])

if test x"$hwloc_header_happy"  = x"yes" -a\
        x"$hwloc_lib_happy"     = x"yes" -a\
	x"$hwloc_version_happy" = x"yes"; then
   AC_DEFINE([HAVE_HWLOC], [1], "hwloc library with ABI > 2.0 is installed.")
fi
AM_CONDITIONAL([HAVE_HWLOC], [test x"$hwloc_header_happy"  = x"yes" -a\
                                   x"$hwloc_lib_happy"     = x"yes" -a\
                                   x"$hwloc_version_happy" = x"yes"])

# excit iterators
PKG_CHECK_MODULES([EXCIT],[libexcit])

# internal jemalloc
ac_configure_args="$ac_configure_args \
			'--with-jemalloc-prefix=jemk_aml_' \
			'--with-install-suffix=-aml'"

AC_CONFIG_HEADERS([include/config.h])

AC_CONFIG_FILES([Makefile
		 src/Makefile
		 src/area/Makefile
		 src/dma/Makefile
		 src/tiling/Makefile
		 src/scratch/Makefile
		 src/utils/Makefile
		 include/Makefile
		 tests/Makefile
		 benchmarks/Makefile
		 aml.pc])
AC_OUTPUT
