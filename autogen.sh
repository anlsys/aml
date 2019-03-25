#!/bin/sh

mkdir -p build-aux
aclocal -I m4
autoreconf -fi

