AML: Building Blocks for Explicit Memory Management
===================================================

AML is a memory management library designed to ease the use of complex memory
topologies and complex data layout optimizations for high-performance computing
applications.

AML is Open Source, distributed under the BSD 3-clause license.

Overview
--------

AML is a framework providing locality-preserving abstractions to application
developers. In particular, AML aims to expose flexible interfaces to describe
and reason about how applications deal with data layout, tiling of data,
placement of data across hardware topologies, and affinity between work and
data.

AML is organized as a collection of abstractions, presented as *building
blocks*, used to develop explicit memory and data management policies. The goals
of AML are:

* **composability**: application developers and performance experts should be
  able to pick and choose the building blocks to use depending on their specific
  needs.

* **flexibility**: users should be able to customize, replace, or change the
  configuration of each building block as much as possible.

AML currently implements the following abstractions:

.. image:: img/building-blocks-diagram.png
   :width: 300px
   :align: right

* :doc:`Area <pages/areas>`, a set of addressable physical memories,
* :doc:`Layout <pages/layout>`, a description of data structure organization,
* :doc:`Tiling <pages/tilings>`, a description of data blocking (decomposition)
* :doc:`DMA <pages/dmas>`, an engine to asynchronously move data structures between areas,
* :doc:`Scratchpad <pages/scratchs>`, a stage-in, stage-out abstraction for prefetching.

Each of these abstractions has several implementations. For instance, areas
may refer to the usual DRAM or its subset, to GPU memory, or to non-volatile memory.
Tilings are implemented to reflect either 1D or 2D structures, and so on.

Quick Start Guide
-----------------

Download
~~~~~~~~

* `Development version <https://xgitlab.cels.anl.gov/argo/aml>`_:

.. code-block:: console
  
  $ git clone git@xgitlab.cels.anl.gov:argo/aml.git

Requirements:
~~~~~~~~~~~~~

* autoconf
* automake
* libtool
* libnuma

Installation
~~~~~~~~~~~~

.. code-block:: console
 
  $ sh autogen.sh
  $ ./configure
  $ make -j install


Workflow
~~~~~~~~

Include the AML header:

.. code-block:: c
  
  #include <aml.h>
  ...
  int main(int argc, char **argv){
  
Check the AML version:

.. code-block:: c
  
  if(aml_version_major != AML_VERSION_MAJOR){
      fprintf(stderr, "AML ABI mismatch!\n");
      return 1;
  }

Initialize and clean up the AML:

.. code-block:: c
  
  if(aml_init(&argc, &argv) != 0){
    fprintf(stderr, "AML library init failure!\n");
    return 1;
  }
  ...
  aml_finalize();

Link your program with *-laml*.

Check the above building-blocks-specific pages for further examples and
information on the library features.

Support
-------

Support for AML is provided through the
`gitlab issues interface <https://xgitlab.cels.anl.gov/argo/aml/issues>`_.
Alternatively you can contact directly the developers/maintainers:

* Swann Perarnau (swann AT anl DOT gov)
* Nicolas Denoyelle (ndenoyelle AT anl DOT gov)

Contributing
------------

AML welcomes comments, suggestions, bug reports, or feature requests, as
well as code contributions. See the
`contributing doc <https://xgitlab.cels.anl.gov/argo/aml/blob/master/CONTRIBUTING.markdown>`_
for more info.

.. toctree::
   :maxdepth: 2
   :caption: Table of Contents

   tutorials/tutorials
   pages/areas
   pages/tilings
   pages/layout
   pages/dmas
   pages/scratchs
