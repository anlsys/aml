# see semver.org for version info
AC_INIT([aml],[0.0.1],[swann@anl.gov])

# are we in the right source dir ?
AC_CONFIG_SRCDIR([src/aml.h])

# build artefacts in separate dir
AC_CONFIG_AUX_DIR([m4])
AC_CONFIG_MACRO_DIR([m4])

# automake should fail on any error
AM_INIT_AUTOMAKE([-Wall -Werror foreign 1.12])
AM_PROG_AR

# check for libtool
LT_INIT

# check for programs
AC_LANG([C])
AC_PROG_CC
AC_PROG_CC_STDC
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
	AX_OPENMP([AC_SUBST(OPENMP_FLAG, $OPENMP_CFLAGS)])
	AC_CHECK_LIB(m, sqrt)
fi
AM_CONDITIONAL([ADD_BENCHMARKS],[test "x$benchmarks" = xtrue])


# dependencies
AC_CHECK_HEADERS(numa.h)
AC_CHECK_LIB(numa, move_pages)

# internal jemalloc
ac_configure_args="$ac_configure_args \
			'--with-jemalloc-prefix=jemk_' \
			'--with-install-suffix=-aml'"
AC_CONFIG_SUBDIRS([jemalloc])

AC_CONFIG_HEADERS([src/config.h])

AC_CONFIG_FILES([Makefile
		 src/Makefile
		 tests/Makefile
		 benchmarks/Makefile
		 aml.pc])
AC_OUTPUT
