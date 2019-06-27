# See semver.org for version info
#################################

m4_define([VERSION_MAJOR], [0])
m4_define([VERSION_MINOR], [1])
m4_define([VERSION_PATCH], [0])
m4_define([VERSION_STRING], VERSION_MAJOR.VERSION_MINOR.VERSION_PATCH)

# Init build tools
##################

AC_INIT([aml],[VERSION_STRING],[swann@anl.gov])
AC_CONFIG_SRCDIR([include/aml.h])
AC_CONFIG_AUX_DIR([m4])
AC_CONFIG_MACRO_DIR([m4])
# automake should fail on any error
AM_INIT_AUTOMAKE([-Wall -Werror foreign subdir-objects 1.12])

# Detect features
#################

AC_LANG([C])
AC_USE_SYSTEM_EXTENSIONS
AC_PROG_CC
AC_PROG_CC_C99
AM_PROG_CC_C_O
AC_PROG_CPP
AC_TYPE_SIZE_T
AC_TYPE_INTPTR_T

AM_PROG_AR
LT_INIT

# Extra dependencies, configuration
###################################

AC_SUBST([PACKAGE_VERSION_MAJOR],[VERSION_MAJOR])
AC_SUBST([PACKAGE_VERSION_MINOR],[VERSION_MINOR])
AC_SUBST([PACKAGE_VERSION_PATCH],[VERSION_PATCH])


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

AC_CHECK_LIB(dl, dlopen)

# add pthread support.
# doc in m4/ax_pthread.m4. Defines automake PTHREAD_CFLAGS and PTHREAD_LIBS
AX_PTHREAD
CC="$PTHREAD_CC"

# NUMA support
AC_CHECK_HEADERS([numa.h],,[AC_MSG_ERROR([AML requires libnuma headers.])])
AC_CHECK_HEADERS([numaif.h],,[AC_MSG_ERROR([AML requires libnuma headers.])])
AC_CHECK_LIB(numa, mbind,,[AC_MSG_ERROR([AML requires libnuma.])])


# check doxygen + sphinx for documentation build
AC_CHECK_PROG([DOXYGEN], [doxygen], [doxygen], [no])
AC_CHECK_PROG([SPHINXBUILD], [sphinx-build], [sphinx-build], [no])
if [[ "x$DOXYGEN" != xno ]]; then
	if [[ "x$SPHINXBUILD" != xno ]]; then
		AC_MSG_NOTICE([Doxygen and Sphinx found, documentation will be build])
		BUILD_DOCS=yes
	else
		AC_MSG_NOTICE([Sphinx not found, cannot build documentation])
		BUILD_DOCS=no
	fi
else
	AC_MSG_NOTICE([Doxygen not found, cannot build documentation])
	BUILD_DOCS=no
fi
AM_CONDITIONAL([BUILD_DOCS],[ test "x$BUILD_DOCS" = xyes ])

# Output
########

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
		 doc/Makefile
		 benchmarks/Makefile
		 aml.pc
		 include/aml/utils/version.h])
AC_OUTPUT
