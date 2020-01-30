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
AC_PROG_AWK
AC_PROG_GREP
AM_PROG_AR
LT_INIT

# Extra dependencies, configuration
###################################

AC_SUBST([PACKAGE_VERSION_MAJOR],[VERSION_MAJOR])
AC_SUBST([PACKAGE_VERSION_MINOR],[VERSION_MINOR])
AC_SUBST([PACKAGE_VERSION_PATCH],[VERSION_PATCH])

# support for testing with valgrind
###################################

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
##################################

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
######################

# doc in m4/ax_pthread.m4. Defines automake PTHREAD_CFLAGS and PTHREAD_LIBS
AX_PTHREAD([],[AC_MSG_ERROR([Cannot find how to compile with pthreads.])])
CC="$PTHREAD_CC"

# NUMA support
##############

AC_CHECK_HEADERS([numa.h],,[AC_MSG_ERROR([AML requires libnuma headers.])])
AC_CHECK_HEADERS([numaif.h],,[AC_MSG_ERROR([AML requires libnuma headers.])])
AC_CHECK_LIB(numa, mbind,,[AC_MSG_ERROR([AML requires libnuma.])])

# check doxygen + sphinx for documentation build
################################################

AC_ARG_ENABLE(docs,
[AS_HELP_STRING([--enable-docs],
		[Generate full html documentation (default is no).])],
[docs=true],[docs=false])

if [[ "x$docs" = xtrue ]]; then
	AC_CHECK_PROG([DOXYGEN], [doxygen], [doxygen], [no])
	if [[ "x$DOXYGEN" == xno ]]; then
		AC_MSG_ERROR([Doxygen not found])
	fi

	AC_CHECK_PROG([SPHINXBUILD], [sphinx-build], [sphinx-build], [no])
	if [[ "x$SPHINXBUILD" == xno ]]; then
		AC_MSG_ERROR([Sphinx not found])
	fi
fi
AM_CONDITIONAL([BUILD_DOCS],[ test "x$docs" == xtrue ])

# check CUDA compiler and libraries
#####################################

have_cuda=0

AC_ARG_WITH([cuda],
	[AS_HELP_STRING([--with-cuda@<:@=yes|no|DIR@:>@],
		[support cuda inside the library (default is check)])],
	[
	if [[ "x$withval" = xno ]]; then
		want_cuda="no"
	elif [[ "x$withval" = xyes ]]; then
		want_cuda="yes"
		cuda_home_path="$CUDA_HOME"
	else
		want_cuda="yes"
		cuda_home_path=$withval
	fi
	],
	[
		want_cuda="check"
		cuda_home_path="$CUDA_HOME"
	])

if [[ "x$want_cuda" != xno ]]; then

	AC_MSG_NOTICE([starting checks for CUDA])
	if [[ -n "$cuda_home_path" ]]; then
		nvcc_search_dirs="$PATH$PATH_SEPARATOR$cuda_home_path/bin"
	else
		nvcc_search_dirs="$PATH"
	fi

	AC_PATH_PROG([NVCC], [nvcc], [], [$nvcc_search_dirs])
	if [[ -n "$NVCC" ]]; then
		have_nvcc="yes"
	fi
else
	AC_MSG_NOTICE([will not check for CUDA])
fi

if [[ "x$have_nvcc" = xyes ]]; then

	if [[ -n "$cuda_home_path" ]]; then
		CUDA_CFLAGS="-I$cuda_home_path/include"
		CUDA_LIBS="-L$cuda_home_path/lib64 -lcudart"
	else
		CUDA_CFLAGS="-I/usr/local/cuda/include"
		CUDA_LIBS="-L/usr/local/cuda/lib64 -lcudart"
	fi
	saved_LIBS=$LIBS
	saved_CFLAGS=$CFLAGS
	LIBS="$LIBS $CUDA_LIBS"
	CFLAGS="$CFLAGS $CUDA_CFLAGS"
	AC_CHECK_HEADER([cuda.h],,
			[AC_MSG_ERROR([could not find cuda.h])])
	AC_CHECK_HEADER([cuda_runtime.h],,
			[AC_MSG_ERROR([could not find cuda_runtime.h])])
	AC_CHECK_LIB(cudart, cudaMalloc,,
		     AC_MSG_ERROR([could not find cudart library]))
	LIBS=$saved_LIBS
	CFLAGS=$saved_CFLAGS
	have_cuda=1
fi

AC_DEFINE_UNQUOTED([HAVE_CUDA], [$have_cuda], [Whether aml support cuda library calls.])
AC_SUBST([HAVE_CUDA],[$have_cuda])
AM_CONDITIONAL([HAVE_CUDA], [ test "$have_cuda" == "1" ])
AC_SUBST(CUDA_CFLAGS)
AC_SUBST(CUDA_LIBS)
AC_SUBST(NVCC)

# Output
########

AC_CONFIG_HEADERS([include/config.h])

AC_CONFIG_FILES([Makefile
		 src/Makefile
		 include/Makefile
		 tests/Makefile
		 tutorials/Makefile		 
		 doc/Makefile
		 benchmarks/Makefile
		 aml.pc
		 include/aml/utils/version.h
		 include/aml/utils/features.h])
AC_CONFIG_FILES([o2lo], [chmod +x o2lo])
AC_OUTPUT
