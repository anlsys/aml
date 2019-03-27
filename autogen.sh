#!/bin/sh
set -ex

# aml
mkdir -p build-aux
aclocal -I m4
autoreconf -fi
