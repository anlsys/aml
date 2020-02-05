DMAs
====

What is an AML DMA ?
--------------------

In computer science, DMA (Direct Memory Access) is an hardware accelerated
method for moving data across memory regions without the intervention of a
compute unit.

Similarly, in AML a DMA is the building block used to move data. Data is
generally moved between two `areas <../../pages/areas>`_, 
between two virtual memory ranges represented by a pointer.  Data is moved from
one (layout)[https://argo-aml.readthedocs.io/en/latest/pages/layout.html] to
another, i.e data can be reorganized during the DMA request.  Data can also be
moved asynchronously by a DMA engine, e.g pthreads on a CPU and cuda streams on
cuda accelerators.

The API for using AML DMA is broke down into two levels.
* The high-level API in the (main header)[../../include/aml.h] provides
generic functions that can be applied on all DMAs.
It also describes the general structure of a DMA for implementers.
* Implementations specific methods, constructors and static DMAs declarations
stand in the second level of headers (aml/dma/\*.h)[../../include/aml/dma].

Examples of AML DMA use cases
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

How to use AML DMA ?
--------------------

Default Areas
-------------

Allocating Memory From an Area
------------------------------

Exercise: allocating interleaved pages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CUDA Area
---------

Exercise: CUDA allocation
~~~~~~~~~~~~~~~~~~~~~~~~~


Implementing a Custom Area
--------------------------

Exercise: interleaving in blocks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
