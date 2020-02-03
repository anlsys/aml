#!/usr/bin/env bash

# -----------------------------------------------------------
# Copyright 2019 UChicago Argonne, LLC.
# (c.f. AUTHORS, LICENSE)
#
# This file is part of the AML project.
# For more info, see https://xgitlab.cels.anl.gov/argo/aml
#
# SPDX-License-Identifier: BSD-3-Clause
# -----------------------------------------------------------

CFLAGS="-Wall -Wextra -Werror -pedantic -O0 -g"
LDFLAGS="-lnuma -laml -lpthread -lcudart"

if [ "x$1" == "xclean" ]; then
		rm -f \
			 0_example \
			 1_reduction
else
# Compile tutorial 0
gcc $CFLAGS 0_example.c -o 0_example $LDFLAGS

# Compile tutorial 1
gcc $CFLAGS 1_reduction.c -o 1_reduction $LDFLAGS
fi
