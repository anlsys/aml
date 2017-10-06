#!/bin/sh
set -ex
mkdir -p build-aux
aclocal -I m4
libtoolize
automake --add-missing --copy
autoconf
