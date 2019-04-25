Welcome to AML Documentation Page {#index}
============

AML is a memory management library designed to ease the use of complex memory
topologies and complex data layout optimizations for high-performance computing
applications.

AML is Open Source, distributed under the BSD 3 clause license.

## Overview

AML is a framework providing locality-preserving abstractions to application
developers. In particular, AML aims to expose flexible interfaces to describe
and reason about how applications deal with data layout, tiling of data,
placement of data across hardware topologies, and affinity between work and
data.

AML is organized as a collection of abstractions, presented as *building
blocks*, to develop explicit memory and data management policies. AML goals
are:
* __composability__: application developers and performance experts should be
to pick and choose which building blocks to use depending on their specific
needs.
* __flexibility__: users should be able to customize, replace, or change the
configuration of each building block as much as possible.

As of now, AML implements the following abstractions:

\image html building-blocks-diagram.png width=400cm

* [Areas](@ref aml_area), a set of addressable physical memories,
* [Tilings](@ref aml_tiling), a description of tiling data structures,
* [DMA](@ref aml_dma), an engine to asynchronously move data structures between areas,
* [Scratchpad](@ref aml_scratch), a stage-in, stage-out abstraction for prefetching.

Each of these abstractions have several implementations. For instance, areas
may refer to usual DRAM or a subset of them, GPU memory or non-volatile memory.
Tilings are implemented to reflect either 1D or 2D structures and so on.

## Quick Start Guide

### Download

* [Development version](https://xgitlab.cels.anl.gov/argo/aml):
```
git clone git@xgitlab.cels.anl.gov:argo/aml.git
```
* Release versions:
TODO after release.

### Requirements:

* autoconf
* automake
* libtool
* libnuma

### Installation

```
sh autogen.sh
./configure
make -j install
```

### Workflow

* Include aml header:
```
#include <aml.h>
...
int main(int argc, char **argv){
```

* Check AML version:
```
if(aml_version_major != AML_VERSION_MAJOR){
    printf("AML ABI mismatch!");
    return 1;
}
```

* Initialize and Cleanup AML:
```
if(aml_init(&argc, &argv) != 0){
    printf("AML library init failure!");
    return 1;
}
...
aml_finalize();
```

* Link your program with `-laml`

See above building blocks specific pages for further examples and information
on library features.

## Support

Support for AML is provided through the
[gitlab issue interface](https://xgitlab.cels.anl.gov/argo/aml/issues).
Alternatively you can contact directly the developers/maintainers:
* Swann Perarnau (swann AT anl DOT gov)
* Nicolas Denoyelle (ndenoyelle AT anl DOT gov)

## Contributing

AML welcomes any comment, suggestion, bug reporting, or feature request, as
well as code contributions. See the [contributing page](@ref contributing_page)
for more info.

