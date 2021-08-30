# See semver.org for version info
#################################

m4_define([VERSION_STRING], m4_esyscmd([build-aux/git-version-gen --fallback 0.1.0.0 .tarball-version]))
m4_define([VERSION_FIELDS], m4_split(VERSION_STRING, [\.]))
m4_define([VERSION_MAJOR], m4_argn(1, VERSION_FIELDS))
m4_define([VERSION_MINOR], m4_argn(2, VERSION_FIELDS))
m4_define([VERSION_PATCH], m4_argn(3, VERSION_FIELDS))
m4_define([VERSION_REVISION], m4_argn(4, VERSION_FIELDS))

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
AC_SUBST([PACKAGE_VERSION_REVISION],[VERSION_REVISION])

# support for testing with valgrind
###################################

AX_VALGRIND_DFLT([helgrind], [off])
AX_VALGRIND_DFLT([drd], [off])
AX_VALGRIND_DFLT([sgcheck], [off])
AX_VALGRIND_CHECK

AC_CHECK_HEADERS([math.h],,[AC_MSG_ERROR([AML requires libmath headers.])])
AC_CHECK_LIB(m, sqrt,,[AC_MSG_ERROR([AML requires libmath.])])

AC_CHECK_LIB(dl, dlopen)

# add pthread support.
######################

# doc in m4/ax_pthread.m4. Defines automake PTHREAD_CFLAGS and PTHREAD_LIBS
AX_PTHREAD([],[AC_MSG_ERROR([Cannot find how to compile with pthreads.])])
CC="$PTHREAD_CC"

# OpenMP support.
######################

AC_OPENMP

# Allow the user to pass additional openmp flags such as:
# [-fiopenmp -fopenmp-targets=spir64] from LLVM compiler. We test that the
# compiler can actually use user flags before moving forward with the
# compilation.
AC_ARG_WITH([openmp-flags],
	[AS_HELP_STRING([--with-openmp-flags@<:@=CFLAGS@:>@],
		        [Additional OpenMP flags to pass to the compiler.])],
	[save_CFLAGS=$CFLAGS
   	 CFLAGS="$CFLAGS $OPENMP_CFLAGS $withval"
   	 AC_COMPILE_IFELSE([AC_LANG_PROGRAM([#include<omp.h>], [omp_get_num_threads();])],
                           [OPENMP_CFLAGS="$OPENMP_CFLAGS $withval"],
			   [AC_MSG_ERROR([Unsupported OpenMP flags: $withval])])
         CFLAGS="$save_CFLAGS"])

# Intel implementation of OpenMP requires to use the same contexts as
# AML in order to work with libze_loader device pointers obtained with AML
# areas. AML can use the same contexts as OpenMP via the use of
# `omp_target_get_context()`. This function is specific to intel
# implementation of OpenMP and linked from libomptarget.so when the
# application uses the LLVM flag `-fiopenmp -fopenmp-targets=`.
# We define the compile time macro `HAVE_ZE_OMP_CONTEXT` that tells AML
# whether the function `omp_target_get_context()` is defined and is going
# to provide OpenMP contexts or if AML has to create its own contexts.
save_CFLAGS=$CFLAGS
CFLAGS="$CFLAGS $OPENMP_CFLAGS $withval"
AC_RUN_IFELSE([AC_LANG_PROGRAM([#include<omp.h>], [omp_target_get_context(0);])],
              [have_omptarget=1],
              [have_omptarget=0])
CFLAGS="$save_CFLAGS"
AC_DEFINE_UNQUOTED([HAVE_ZE_OMP_CONTEXT],
                   [$have_omptarget],
		   [Whether AML Ze device pointers can be used with Intel OpenMP regions.])

# Report OpenMP support.
if test "x$OPENMP_CFLAGS" = "x"; then
   HAVE_OPENMP="0"
else
   HAVE_OPENMP="1"
fi

# NUMA support
##############

AC_CHECK_HEADERS([numa.h],,[AC_MSG_ERROR([AML requires libnuma headers.])])
AC_CHECK_HEADERS([numaif.h],,[AC_MSG_ERROR([AML requires libnuma headers.])])
AC_CHECK_LIB(numa, mbind,,[AC_MSG_ERROR([AML requires libnuma.])])

# Hwloc support
###############

PKG_CHECK_MODULES([HWLOC], \
                  [hwloc >= 2.1],   \
                  [have_hwloc=1], \
                  [have_hwloc=0])
if test "$have_hwloc" = "1"; then
   AC_DEFINE([HAVE_HWLOC], [1], "hwloc library with ABI > 2.0 is installed.")
fi
AM_CONDITIONAL([HAVE_HWLOC], [test "$have_hwloc" = "1"])
AC_DEFINE_UNQUOTED([HAVE_HWLOC], [$have_hwloc], [Whether aml support hwloc library calls.])
AC_SUBST([HAVE_HWLOC],[$have_hwloc])

# OpenCL support
###############

PKG_CHECK_MODULES([OPENCL], \
                  [OpenCL >= 2.1],   \
                  [have_opencl=1], \
                  [have_opencl=0])
if test "$have_opencl" = "1"; then
   AC_DEFINE([HAVE_OPENCL], [1], "OpenCL library with ABI >= 2.1 is installed.")
fi
AM_CONDITIONAL([HAVE_OPENCL], [test "$have_opencl" = "1"])
AC_DEFINE_UNQUOTED([HAVE_OPENCL], [$have_opencl], [Whether aml support opencl library calls.])
AC_SUBST([HAVE_OPENCL],[$have_opencl])

# Level Zero Support
####################

# PKG CONFIG detection
PKG_CHECK_MODULES([ZE], \
                  [libze_loader >= 1.0],   \
                  [have_ze=1], \
                  [have_ze=0])

# fallback detection
if test "$have_ze" = "0"; then
have_libze=0 # library check
have_ze_header=0 # header check
have_ze_version="no" # version check

# library check
AC_SEARCH_LIBS([zeInit], [ze_loader], [have_libze=1], [have_libze=0])

# header check
if test "$have_libze" = "1"; then
   AC_CHECK_HEADER([level_zero/ze_api.h], [have_ze_header=1], [have_ze_header=0])
fi

# header version check
if test "$have_ze_header" = "1"; then
   AC_MSG_CHECKING([ze_api version >= 1.0])
   AC_RUN_IFELSE([AC_LANG_PROGRAM(
[[#include<level_zero/ze_api.h>]],
[[return ZE_API_VERSION_CURRENT < ZE_MAKE_VERSION( 1, 0 )]]
)],
                 [have_ze_version=yes],
		 [have_ze_version=no])
   AC_MSG_RESULT([$have_ze_version])
fi

# All check passed ->
if test "$have_ze_version" = "yes"; then
   ZE_CFLAGS=""
   ZE_LIBS="$ac_cv_search_zeInit"
   AC_SUBST(ZE_CFLAGS)
   AC_SUBST(ZE_LIBS)
   have_ze=1
fi
fi # test "$have_ze" = "0"

# If level zero is present and usable then declare it as such.
if test "$have_ze" = "1"; then
   AC_DEFINE([HAVE_ZE], [1], "Level Zero library and headers are available")
   if test "$have_omptarget" = "1"; then
     INTEL_OMP_ZE_BACKEND_INTEROPERABILITY_MSG="OPENMP INTEROPERABILITY: yes\
Make sure to enable OpenMP LEVEL0 backend with: export LIBOMPTARGET_PLUGIN=LEVEL0
"
   else
     INTEL_OMP_ZE_BACKEND_INTEROPERABILITY_MSG="OPENMP INTEROPERABILITY: no
Try using: ./configure CC=icx --with-openmp-flags=\"-fiopenmp -fopenmp-targets=spir64\"
     "
   fi
fi
AM_CONDITIONAL([HAVE_ZE], [test "$have_ze" = "1"])
AC_DEFINE_UNQUOTED([HAVE_ZE], [$have_ze], [Whether aml support intel level zero library calls.])
AC_SUBST([HAVE_ZE],[$have_ze])

# check doxygen + sphinx for documentation build
################################################

AC_ARG_ENABLE(docs,
[AS_HELP_STRING([--enable-docs],
		[Generate full html documentation (default is no).])],
[docs=true],[docs=false])

if test "x$docs" = "xtrue"; then
	AC_CHECK_PROG([DOXYGEN], [doxygen], [doxygen], [no])
	if test "x$DOXYGEN" = "xno"; then
		AC_MSG_ERROR([Doxygen not found])
	fi

	AC_CHECK_PROG([SPHINXBUILD], [sphinx-build], [sphinx-build], [no])
	if test "x$SPHINXBUILD" = "xno"; then
		AC_MSG_ERROR([Sphinx not found])
	fi
fi
AM_CONDITIONAL([BUILD_DOCS],[ test "x$docs" = "xtrue" ])

# check CUDA compiler and libraries
#####################################

have_cuda=0

AC_ARG_WITH([cuda],
	[AS_HELP_STRING([--with-cuda@<:@=yes|no|DIR@:>@],
		[support cuda inside the library (default is check)])],
	[
	if test "x$withval" = "xno"; then
		want_cuda="no"
	elif test "x$withval" = "xyes"; then
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

if test "x$want_cuda" != "xno"; then

	AC_MSG_NOTICE([starting checks for CUDA])
	if test -n "$cuda_home_path"; then
		nvcc_search_dirs="$PATH$PATH_SEPARATOR$cuda_home_path/bin"
	else
		nvcc_search_dirs="$PATH"
	fi

	AC_PATH_PROG([NVCC], [nvcc], [], [$nvcc_search_dirs])
	if test -n "$NVCC"; then
		have_nvcc="yes"
	fi
else
	AC_MSG_NOTICE([will not check for CUDA])
fi

if test "x$have_nvcc" = "xyes"; then

	if test -n "$cuda_home_path"; then
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
	AC_CHECK_LIB(cudart, cudaLaunchHostFunc,,
		     AC_MSG_ERROR([could not find cudart library]))
	LIBS=$saved_LIBS
	CFLAGS=$saved_CFLAGS
	have_cuda=1
fi

AC_DEFINE_UNQUOTED([HAVE_CUDA], [$have_cuda], [Whether aml support cuda library calls.])
AC_SUBST([HAVE_CUDA],[$have_cuda])
AM_CONDITIONAL([HAVE_CUDA], [ test "$have_cuda" = "1" ])
AC_SUBST(CUDA_CFLAGS)
AC_SUBST(CUDA_LIBS)
AC_SUBST(NVCC)

# Support for cross-compiling check programs
AM_EXTRA_RECURSIVE_TARGETS([check-programs])

# Output
########

AC_CONFIG_SUBDIRS([excit])

AC_CONFIG_HEADERS([include/config.h])

AC_CONFIG_FILES([Makefile
		 src/Makefile
		 include/Makefile
		 tests/Makefile
		 doc/tutorials/Makefile
		 doc/Makefile
		 doc/tutorials/hello_world/Makefile
		 doc/tutorials/area/Makefile
		 doc/tutorials/dma/Makefile
		 doc/tutorials/layouts/Makefile
		 benchmarks/Makefile
		 aml.pc
		 include/aml/utils/version.h
		 include/aml/utils/features.h])
AC_OUTPUT

# Print out what was configured
cat <<EOF

-------------------------------------------------------------------------------
AML

Version: $PACKAGE_VERSION
Docs:    $docs

FEATURES:
---------

OPENMP:
======

Active:  $HAVE_OPENMP
CFLAGS:  $OPENMP_CFLAGS

HWLOC:
======

Active:  $HAVE_HWLOC
CFLAGS:  $HWLOC_CFLAGS
LDFLAGS: $HWLOC_LIBS

CUDA:
=====

Active:  $HAVE_CUDA
CFLAGS:  $CUDA_CFLAGS
LDFLAGS: $CUDA_LIBS

OpenCL:
=======

Active:  $HAVE_OPENCL
CFLAGS:  $OPENCL_CFLAGS
LDFLAGS: $OPENCL_LIBS

ZE:
===

Active:  $HAVE_ZE
CFLAGS:  $ZE_CFLAGS
LDFLAGS: $ZE_LIBS
$INTEL_OMP_ZE_BACKEND_INTEROPERABILITY_MSG

-------------------------------------------------------------------------------
EOF
