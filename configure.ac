# See semver.org for version info
#################################

m4_define([VERSION_STRING], m4_esyscmd([build-aux/git-version-gen .tarball-version]))
m4_define([VERSION_FIELDS], m4_split(VERSION_STRING, [\.]))
m4_define([VERSION_MAJOR], m4_argn(1, VERSION_FIELDS))
m4_define([VERSION_MINOR], m4_argn(2, VERSION_FIELDS))
m4_define([VERSION_PATCH], m4_argn(3, VERSION_FIELDS))
m4_define([VERSION_REVISION], m4_argn(4, VERSION_FIELDS))

# Utils functions
##################

dnl AML_HAVE_DEFINE_WITH_MODULES(VARIABLE-PREFIX, MODULES,
dnl   [DESCRIPTION], [DEFAULT])
dnl ------------------------------------------------------
dnl
dnl Convenience macro to run AM_CONDITIONAL and AC_DEFINE after
dnl PKG_WITH_MODULES check. HAVE_[VARIABLE-PREFIX] is exported as make
dnl and preprocessor variable.
AC_DEFUN([AML_HAVE_DEFINE_WITH_MODULES],
[
AS_TR_SH([have_]m4_tolower([$1]))=0
PKG_HAVE_WITH_MODULES([$1],[$2],[$3],[$4])
if test "$AS_TR_SH([with_]m4_tolower([$1]))" = "yes"; then
   AS_TR_SH([have_]m4_tolower([$1]))=1
fi
AC_DEFINE_UNQUOTED([HAVE_][$1], $AS_TR_SH([have_]m4_tolower([$1])), [Supports ]m4_tolower([$1]))
AM_CONDITIONAL([HAVE_][$1],[ test $AS_TR_SH([have_]m4_tolower([$1])) = 1])
AC_SUBST([HAVE_][$1],[$[have_]m4_tolower([$1])])
])dnl PKG_HAVE_DEFINE_WITH_MODULES

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

# protext OPENMP_CFLAGS from libtool: libtool filters linker flags it doesn't
# know about, which can easily happen with these ones
if test -n "$OPENMP_CFLAGS"; then
    OPENMP_LIBS=" $OPENMP_CFLAGS"
    # m4 quoting rules make the sed regexp a nightmare
    # use quadrigraph to write this rule: s/[ ][ ]*\([^ ]\)/ -XCClinker \1/g
    OPENMP_LIBS="`AS_ECHO_N("$OPENMP_LIBS") | sed 's/@<:@ @:>@@<:@ @:>@*\(@<:@^ @:>@\)/ -XCClinker \1/g'`"
else
    OPENMP_LIBS=""
fi
AC_SUBST(OPENMP_LIBS)

# MPI support
#############

AX_MPI(have_mpi=1,have_mpi=0)
TESTCC="$MPICC"
AC_SUBST(TESTCC)
AC_SUBST([HAVE_MPI],[$have_mpi])
AC_DEFINE_UNQUOTED([HAVE_MPI], [$have_mpi], [Whether aml supports MPI.])
AM_CONDITIONAL([HAVE_MPI], [test "$have_mpi" == "1"])

# NUMA support
##############

AC_CHECK_HEADERS([numa.h],,[AC_MSG_ERROR([AML requires libnuma headers.])])
AC_CHECK_HEADERS([numaif.h],,[AC_MSG_ERROR([AML requires libnuma headers.])])
AC_CHECK_LIB(numa, mbind,,[AC_MSG_ERROR([AML requires libnuma.])])

# Hwloc support
###############

AML_HAVE_DEFINE_WITH_MODULES(HWLOC, [hwloc >= 2.1],,)

# OpenCL support
###############

AML_HAVE_DEFINE_WITH_MODULES([OPENCL], [OpenCL >= 2.1],,)

# Level Zero Support
####################

# Define --with-ze option and detect with pkg-config
# De do not export variables yet because we want to try a fallback method if
# this one fails.
have_ze=0
PKG_WITH_MODULES([ZE], [libze_loader >= 1.0], have_ze=1, have_ze=0,,)

# This is the fallback method. It used only if the user requests ze support.
if test "x$with_ze" = "xyes" &&
   test "x$have_ze" = "x0"; then
   # Setup user flags for configure detection.
   saved_LIBS=$LIBS
   saved_CFLAGS=$CFLAGS
   LIBS="$ZE_LIBS $LIBS"
   CFLAGS="$ZE_CFLAGS $CFLAGS"
   # Check header and libs
   AC_CHECK_HEADER([level_zero/ze_api.h], [have_ze_header=1], [have_ze_header=0])
   AC_SEARCH_LIBS([zeInit], [ze_loader], [have_libze=1], [have_libze=0])
   # Check version
   AC_MSG_CHECKING([ze_api version >= 1.0])
   AC_RUN_IFELSE([AC_LANG_PROGRAM(
     [[#include<level_zero/ze_api.h>]],
     [[return ZE_API_VERSION_CURRENT < ZE_MAKE_VERSION( 1, 0 )]]
     )],
     [have_ze_version=yes],
     [have_ze_version=no])
   AC_MSG_RESULT([$have_ze_version])
   # Restore libs and cflags.
   LIBS="$saved_LIBS"
   CFLAGS="$saved_CFLAGS"

   # If all detection steps succeeded
   if test "$have_ze_header" = "1" &&
      test "$have_libze" = "1" &&
      test "$have_ze_version" = "yes"; then
         ZE_LIBS="$ZE_LIBS $ac_cv_search_zeInit"
      	 have_ze=1
   fi
   HAVE_ZE=$have_ze
fi

# We export flags with what has been detected or not.
AC_SUBST(ZE_CFLAGS)
AC_SUBST(ZE_LIBS)
AC_SUBST([HAVE_ZE],[$have_ze])
AC_DEFINE_UNQUOTED([HAVE_ZE], [$have_ze], "ze library with ABI > 2.0 is installed.")
AM_CONDITIONAL([HAVE_ZE], [test "$have_ze" = "1"])

# If ze backend is used, we checked earlier whether ze pointers can be used
# as openmp target pointers. Here we print a message to the user on whether
# The support has been detected or not, and a hint of what to do if not.
if test "$have_ze" = "1"; then
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
	[AS_HELP_STRING([--with-cuda],
		[support cuda inside the library (default is check)])],
	[
	if test "x$withval" = "xno"; then
		want_cuda="no"
	else
		want_cuda="yes"
	fi
	],
	[
		want_cuda="check"
	])
AC_ARG_VAR(CUDA_HOME, Path where cuda is installed.)

if test "x$want_cuda" != "xno"; then

	AC_MSG_NOTICE([starting checks for CUDA])
	if test -n "$CUDA_HOME"; then
		nvcc_search_dirs="$PATH$PATH_SEPARATOR$CUDA_HOME/bin"
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

	if test -n "$CUDA_HOME"; then
		CUDA_CFLAGS="-I$CUDA_HOME/include"
		CUDA_LIBS="-L$CUDA_HOME/lib64 -lcudart"
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

# check HIP compiler and libraries
#####################################

have_hip=0
HIP_LIBS=""
HIP_CFLAGS=""

# Hip header requires that one of these is defined or else compiler will fail.
# If cuda is available on the platform, we use it as a backend for hip, else
# we use AMD backend.
if test "$have_cuda" == "1"; then
HIP_PLATFORM="__HIP_PLATFORM_NVIDIA__"
else
HIP_PLATFORM="__HIP_PLATFORM_AMD__"
fi

# Let user customize backend.
AC_ARG_WITH([hip-platform],
	[AS_HELP_STRING([--with-hip-platform=@<:@=amd|nvidia@:>@],
	[Set HIP backend.])],
	[if test "x$withval" = "xamd"; then
	    HIP_PLATFORM="__HIP_PLATFORM_AMD__"
	elif test "x$withval" = "xnvidia"; then
	    HIP_PLATFORM="__HIP_PLATFORM_NVIDIA__"
	else
	    AC_MSG_ERROR([Invalid HIP backend (amd|nvidia).])
	fi])
if test "$HIP_PLATFORM" = "__HIP_PLATFORM_NVIDIA__" && test "$have_cuda" == "0"; then
   AC_MSG_ERROR([HIP backend nvidia is not supported without cuda support enabled.])
fi

# Read custom user configuration of hip directory.
AC_ARG_VAR(HIP_PATH, Path where rocm is installed. hip headers and library are searched in HIP_PATH.)

AC_ARG_WITH([hip],
   AS_HELP_STRING([--with-hip, Enable support for hip backend. Default is check in HIP_PATH"]),
   [if test "x$withval" = "xno"; then
      with_hip="no"
   elif test "x$withval" = "xyes"; then
      with_hip="yes"
   fi],
   with_hip="check")

if test "$with_hip" = "no"; then
   AC_MSG_NOTICE([Will not check for hip.])
else
   # Set new flags and save old flags
   if test -z "$HIP_PATH"; then
      test -z "$ROCM_PATH" && ROCM_PATH=/opt/rocm # set a default value.
      # Apparently $ROCM_PATH/hip is obsolete?
      HIP_PATH=$ROCM_PATH
   fi
   HIP_LIBS="-L$HIP_PATH/lib"
   HIP_CFLAGS="-D$HIP_PLATFORM -I$HIP_PATH/include"
   saved_LIBS=$LIBS
   saved_CFLAGS=$CFLAGS
   LIBS="$LIBS $HIP_LIBS"
   CFLAGS="$CFLAGS $HIP_CFLAGS"

   # Check libs and headers.
   AC_SEARCH_LIBS([hipDeviceGet], [amdhip64], [have_hip_lib="yes"],[have_hip_lib="no"])
   test "$have_hip_lib" = "yes" && HIP_LIBS="$HIP_LIBS $ac_cv_search_hipDeviceGet"

   # The preprocessor fails but not the compiler because the preprocessor does not
   # have the required define "__HIP_PLATFORM_*" passed through CFLAGS.
   # As a result, the header is found but a warning is displayed.
   AC_CHECK_HEADERS([hip/hip_runtime_api.h],[have_hip_header="yes"],[have_hip_header="no"])

   CFLAGS=$saved_CFLAGS
   LIBS=$saved_LIBS

   # Match user requirements with what was found.
   if test "$have_hip_header" = "no" ||
      test "$have_hip_lib" = "no"; then
      HIP_CFLAGS=""
      HIP_LIBS=""
      if test "$with_hip" = "yes"; then
         AC_MSG_ERROR([Missing hip library.])
      fi
   else
      have_hip=1
   fi
fi # "$with_hip" = "no"

# If we are going to use hip, we export defines and automake flags.
# Else we restore autoconf LIBS and CFLAGS.
HAVE_HIP=$have_hip
AM_CONDITIONAL([HAVE_HIP], [ test "$have_hip" = "1" ])
AC_DEFINE_UNQUOTED([HAVE_HIP], [$have_hip], [Whether aml support hip library calls.])
AC_SUBST([HAVE_HIP],[$have_hip])
AC_SUBST(HIP_CFLAGS)
AC_SUBST(HIP_LIBS)

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
LDFLAGS: $OPENMP_LIBS

MPI:
======

Active:  $HAVE_MPI
TESTCC: $TESTCC
LDFLAGS: $MPILIBS

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

HIP:
===

Active:  $HAVE_HIP
CFLAGS:  $HIP_CFLAGS
LDFLAGS: $HIP_LIBS

-------------------------------------------------------------------------------
EOF
