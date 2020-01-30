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

# Compile tutorial 0
gcc $CFLAGS 0_isInterleaved.c -o 0_isInterleaved $LDFLAGS

# Compile tutorial 1
gcc $CFLAGS 1_aml_area_linux.c -o 1_aml_area_linux $LDFLAGS

# Compile tutorial 2
gcc $CFLAGS 2_custom_interleave_area.c -o 2_custom_interleave_area $LDFLAGS

# Compile tutorial 3
gcc $CFLAGS 3_aml_area_cuda.c -o 3_aml_area_cuda $LDFLAGS
