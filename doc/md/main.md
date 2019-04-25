Welcome to AML Documentation Page {#index}
============

AML is a memory management library designed for highly optimized
codes to delegate complex memory operations.

## Overview

AML goals are the following:
1. to provide a convenient interface for addressing all heterogeneous
addressable memories and moving data around.
2. to provide an interface for expressing and translating data
structures between several representations and physical memories.

Toward achieving these goals, AML implements the following abstractions:

\image html building-blocks-diagram.png width=400cm

* [Areas](@ref aml_area), a set of addressable physical memories,
* [Tilings](@ref aml_tiling), a description of tiling data structures,
* [DMA](@ref aml_dma), an engine to asynchronously move data structures between areas,
* [Scratchpad](@ref aml_scratch), a stage-in, stage-out abstraction for prefetching.

Each of these abstractions have several implementations. For instance,
areas may refer to usual DDR of a subset of them, GPU memory or non-volatile memory.
Tilings are implemented to reflect either 1D or 2D structures etc.

## Problems Solved with AML

AML library is designed for highly optimized codes to delegate complex memory
operations. AML is currently used in a
[proof of concept](https://xgitlab.cels.anl.gov/argo/nauts/tree/master/papers/mchpc18)
of efficient matrix multiplication on architectures with software managed memory side
cache. It is also currently being extended toward specialized memory allocations
for latency sensitive applications.

## Quick Start Guide

### Download

* [Github version](https://xgitlab.cels.anl.gov/argo/aml):
```
git clone git@xgitlab.cels.anl.gov:argo/aml.git
```
* Release versions: 
TODO after release.

### Requirements:

* autoconf
* automake
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

See above building blocks specific pages for further examples and information on
library features.

## Support

AML interface with users is mainly done through
[gitlab issue interface](https://xgitlab.cels.anl.gov/argo/aml/issues) and
direct contact with developers/maintainers:
* Swann Perarnau (swann@anl.gov)
* Nicolas Denoyelle (ndenoyelle@anl.gov)

AML is looking forward to new users. If you think your problem can be solved with AML
eventually with some additional non-existing features which are relevant to the library,
please let us know. Tell us about your topic of interests and your scientific/technical
problems. We are also pleased to improve the software, thus any suggestion, bug
declaration, contribution is welcomed.

## Contributing

AML is looking forward to new collaborations and contributions.
If you wish to contribute to AML project rendez-vous [here](@ref contributing_page)

