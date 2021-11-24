# See semver.org for version info
#################################

m4_define([VERSION_STRING], m4_esyscmd([build-aux/git-version-gen .tarball-version]))
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

# The directory where to look for hwloc. If no, we try to detect it automatically.
hwloc_dir=no
# Whether hwloc was successfully detected (0|1). 
have_hwloc=0
# Whether the user wants hwloc: (yes|no|check)
# yes: the package is required
# no: the package should not be checked.
# check: autodetect the package and enable it if it was found.
want_hwloc=check

AC_ARG_WITH([hwloc],
	[AS_HELP_STRING([--with-hwloc@<:@=yes|no|DIR@:>@],
		[Support hwloc backend and features in the library  (default is check)])],
	[if test "x$withval" = "xno"; then
		want_hwloc=no
	elif test "x$withval" = "xyes"; then
	     	want_hwloc=yes
	else
		want_hwloc=yes
		hwloc_dir=$withval
	fi],[want_hwloc="check"])

# The default check is to rely on pkg-config for detection.
AS_IF([test "$want_hwloc" = "check" ||
       test "$want_hwloc" = "yes" &&
       test "$hwloc_dir" = "no"], [
   PKG_CHECK_MODULES([HWLOC], [hwloc >= 2.1], [have_hwloc=1], [have_hwloc=0])
])
if test "$want_hwloc" != "no" &&
   test "$hwloc_dir" != "no"; then
   # The user requires a specific location for where to find hwloc.
   # We use the classic headers and library detection and fail if detection fails.
   HWLOC_CFLAGS=-I$hwloc_dir/include
   HWLOC_LIBS="-L$hwloc_dir/lib"
   saved_LIBS=$LIBS
   saved_CFLAGS=$CFLAGS
   LIBS="$HWLOC_LIBS $LIBS"
   CFLAGS="$HWLOC_CFLAGS $CFLAGS"
   HWLOC_LIBS="$HWLOC_LIBS -lhwloc"
   AC_CHECK_HEADER([hwloc.h],,[AC_MSG_ERROR([hwloc.h header not found.])])
   AC_CHECK_LIB(hwloc, hwloc_topology_init,, AC_MSG_ERROR([could not find hwloc library]))
   LIBS="$saved_LIBS"
   CFLAGS="$saved_CFLAGS"
   have_hwloc=1
   AC_SUBST(HWLOC_CFLAGS)
   AC_SUBST(HWLOC_LIBS)
fi
if test "$want_hwloc" == "yes" &&
   test "$have_hwloc" == "0"; then
   AC_MSG_ERROR([hwloc required but not found.])
fi

# We export substitutions and defines if needed.
if test "$have_hwloc" = "1"; then
   AC_DEFINE([HAVE_HWLOC], [1], "hwloc library with ABI > 2.0 is installed.")
fi
AM_CONDITIONAL([HAVE_HWLOC], [test "$have_hwloc" = "1"])
AC_DEFINE_UNQUOTED([HAVE_HWLOC], [$have_hwloc], [Whether aml support hwloc library calls.])
AC_SUBST([HAVE_HWLOC],[$have_hwloc])

# OpenCL support
###############

# The directory where to look for opencl. If no, we try to detect it automatically.
opencl_dir=no
# Whether opencl was successfully detected (0|1). 
have_opencl=0
# Whether the user wants opencl: (yes|no|check)
# yes: the package is required
# no: the package should not be checked.
# check: autodetect the package and enable it if it was found.
want_opencl=check

AC_ARG_WITH([opencl],
	[AS_HELP_STRING([--with-opencl@<:@=yes|no|DIR@:>@],
		[Support opencl backend and features in the library  (default is check)])],
	[if test "x$withval" = "xno"; then
		want_opencl=no
	elif test "x$withval" = "xyes"; then
	     	want_opencl=yes
	else
		want_opencl=yes
		opencl_dir=$withval
	fi],[want_opencl="check"])

AS_IF([test "$want_opencl" = "check" ||
       test "$want_opencl" = "yes" &&
       test "$opencl_dir" = "no" ], [
   # The default check is to rely on pkg-config for detection.
   PKG_CHECK_MODULES([OPENCL], [OpenCL >= 2.1], [have_opencl=1], [have_opencl=0])
])
if test "$want_opencl" != "no" &&
     test "$opencl_dir" != "no"; then
   # The user requires a specific location for where to find opencl.
   # We use the classic headers and library detection and fail if detection fails.
   OPENCL_CFLAGS=-I$opencl_dir/include
   OPENCL_LIBS="-L$opencl_dir/lib"
   saved_LIBS=$LIBS
   saved_CFLAGS=$CFLAGS
   LIBS="$OPENCL_LIBS $LIBS"
   CFLAGS="$OPENCL_CFLAGS $CFLAGS"
   OPENCL_LIBS="$OPENCL_LIBS -lOpenCL"
   AC_CHECK_HEADER([CL/opencl.h],,[AC_MSG_ERROR([opencl.h header not found.])])
   AC_CHECK_LIB(OpenCL, clGetPlatformIDs,, AC_MSG_ERROR([could not find opencl library]))
   LIBS="$saved_LIBS"
   CFLAGS="$saved_CFLAGS"
   have_opencl=1
   AC_SUBST(OPENCL_CFLAGS)
   AC_SUBST(OPENCL_LIBS)
fi
if test "$want_opencl" == "yes" &&
   test "$have_opencl" == "0"; then
   AC_MSG_ERROR([opencl required but not found.])
fi

# We export substitutions and defines if needed.
if test "$have_opencl" = "1"; then
   AC_DEFINE([HAVE_OPENCL], [1], "opencl library with ABI > 2.0 is installed.")
fi
AM_CONDITIONAL([HAVE_OPENCL], [test "$have_opencl" = "1"])
AC_DEFINE_UNQUOTED([HAVE_OPENCL], [$have_opencl], [Whether aml support opencl library calls.])
AC_SUBST([HAVE_OPENCL],[$have_opencl])

# Level Zero Support
####################

# The directory where to look for ze. If no, we try to detect it automatically.
ze_dir=no
# Whether ze was successfully detected (0|1). 
have_ze=0
# Whether the user wants ze: (yes|no|check)
# yes: the package is required
# no: the package should not be checked.
# check: autodetect the package and enable it if it was found.
want_ze=check

AC_ARG_WITH([ze],
	[AS_HELP_STRING([--with-ze@<:@=yes|no|DIR@:>@],
		[Support ze backend and features in the library  (default is check)])],
	[if test "x$withval" = "xno"; then
		want_ze=no
	elif test "x$withval" = "xyes"; then
	     	want_ze=yes
	else
		want_ze=yes
		ze_dir=$withval
	fi],[want_ze="check"])

AS_IF([test "$want_ze" != "no" &&
       test "$ze_dir" == "no"], [
   # The default check is to rely on pkg-config for detection.
   PKG_CHECK_MODULES([ZE], [libze_loader >= 1.0], [have_ze=1], [have_ze=0])
])
if test "$want_ze" != "no" &&
   test "$have_ze" = "0"; then
   ZE_CFLAGS=""
   ZE_LIBS=""
   if test "$ze_dir" != "no"; then
      # The user requires a specific location for where to find ze.
      ZE_CFLAGS=-I$ze_dir/include
      ZE_LIBS=-L$ze_dir/lib
   fi
   saved_LIBS=$LIBS
   saved_CFLAGS=$CFLAGS
   LIBS="$ZE_LIBS $LIBS"
   CFLAGS="$ZE_CFLAGS $CFLAGS"
   AC_CHECK_HEADER([level_zero/ze_api.h], [have_ze_header=1], [have_ze_header=0])
   AC_SEARCH_LIBS([zeInit], [ze_loader], [have_libze=1], [have_libze=0])
   if test "$have_ze_header" = "1" &&
      test "$have_libze" = "1"; then
     AC_MSG_CHECKING([ze_api version >= 1.0])
     AC_RUN_IFELSE([AC_LANG_PROGRAM(
     [[#include<level_zero/ze_api.h>]],
     [[return ZE_API_VERSION_CURRENT < ZE_MAKE_VERSION( 1, 0 )]]
     )],
     [have_ze_version=yes],
     [have_ze_version=no])
     AC_MSG_RESULT([$have_ze_version])
   fi
   LIBS="$saved_LIBS"
   CFLAGS="$saved_CFLAGS"
   
   if test "$have_ze_header" = "1" &&
      test "$have_libze" = "1" &&
      test "x$have_ze_version" = "xyes"; then
         ZE_LIBS="$ZE_LIBS $ac_cv_search_zeInit"
      	 have_ze=1
   	 AC_SUBST(ZE_CFLAGS)
   	 AC_SUBST(ZE_LIBS)
   fi
fi
if test "$want_ze" == "yes" &&
   test "$have_ze" == "0"; then
   AC_MSG_ERROR([ze required but not found.])
fi

# We export substitutions and defines if needed.
if test "$have_ze" = "1"; then
   AC_DEFINE([HAVE_ZE], [1], "ze library with ABI > 2.0 is installed.")
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
AC_DEFINE_UNQUOTED([HAVE_ZE], [$have_ze], [Whether aml support ze library calls.])
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
		     AC_MSG_ERROR([could not find cudart library]))w
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

AC_CONFIG_HEADERS([include/internal/config.h])

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
