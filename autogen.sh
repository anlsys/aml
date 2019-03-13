#!/bin/sh
set -ex

# aml
mkdir -p build-aux
aclocal -I m4
autoheader
libtoolize
automake --add-missing --copy
autoconf
