AML: Building Blocks for Explicit Memory Management
===================================================

AML is a memory management library designed to ease the use of complex memory
topologies and complex data layout optimizations for high-performance computing
applications.

AML is Open Source, distributed under the BSD 3 clause license.

Overview
--------

AML is a framework providing locality-preserving abstractions to application
developers. In particular, AML aims to expose flexible interfaces to describe
and reason about how applications deal with data layout, tiling of data,
placement of data across hardware topologies, and affinity between work and
data.

AML is organized as a collection of abstractions, presented as *building
blocks*, to develop explicit memory and data management policies. AML goals
are:

* **composability**: application developers and performance experts should be
  to pick and choose which building blocks to use depending on their specific
  needs.

* **flexibility**: users should be able to customize, replace, or change the
  configuration of each building block as much as possible.

As of now, AML implements the following abstractions:

.. image:: img/building-blocks-diagram.png
   :width: 300px
   :align: right

* :doc:`Areas <pages/areas>`, a set of addressable physical memories,
* :doc:`Layout <pages/layout>`, a description of data structures organization,
* :doc:`Tilings <pages/tilings>`, (soon to be replaced),
* :doc:`DMAs <pages/dmas>`, an engine to asynchronously move data structures between areas,
* :doc:`Scratchpads <pages/scratchs>`, a stage-in, stage-out abstraction for prefetching.

Each of these abstractions have several implementations. For instance, areas
may refer to usual DRAM or a subset of them, GPU memory or non-volatile memory.
Tilings are implemented to reflect either 1D or 2D structures and so on.

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

Include aml header:

.. code-block:: c
  
  #include <aml.h>
  ...
  int main(int argc, char **argv){
  
Check AML version:

.. code-block:: c
  
  if(aml_version_major != AML_VERSION_MAJOR){
      printf("AML ABI mismatch!");
      return 1;
  }

Initialize and Cleanup AML:

.. code-block:: c
  
  if(aml_init(&argc, &argv) != 0){
    printf("AML library init failure!");
    return 1;
  }
  ...
  aml_finalize();

Link your program with *-laml*.

See above building blocks specific pages for further examples and information
on library features.

Support
-------

Support for AML is provided through the
`gitlab issue interface <https://xgitlab.cels.anl.gov/argo/aml/issues>`_.
Alternatively you can contact directly the developers/maintainers:

* Swann Perarnau (swann AT anl DOT gov)
* Nicolas Denoyelle (ndenoyelle AT anl DOT gov)

Contributing
------------

AML welcomes any comment, suggestion, bug reporting, or feature request, as
well as code contributions. See the
`contributing doc <https://xgitlab.cels.anl.gov/argo/aml/blob/master/CONTRIBUTING.markdown>`_
for more info.

.. toctree::
   :maxdepth: 2
   :caption: Table of Contents

   pages/areas
   pages/tilings
   pages/layout
   pages/dmas
   pages/scratchs
