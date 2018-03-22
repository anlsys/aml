#!/bin/sh
set -ex

# jemalloc
(cd jemalloc; ./autogen.sh)

# aml
mkdir -p build-aux
aclocal -I m4
autoheader
libtoolize
automake --add-missing --copy
autoconf
