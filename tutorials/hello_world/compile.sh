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
LDFLAGS="-laml"

# Compile hello world tutorial
gcc $CFLAGS 0_hello.c -o 0_hello $LDFLAGS
