Tilings: Decomposing Data
====================================

Tiling is a representation of data structures as arrays. 

An AML tiling structure can be defined as a multi-dimensional grid of data,
like a matrix, a stencil, etc... 
Tilings are used in AML as a description of a macro data structure that will be
used by a library to do its own work.
This structure is exploitable by AML to perform optimized movement operations.

You can think of a tiling as 1D or 2D contiguous array.
The tiles in the structure can be of custom size and AML provides iterators to
easily access tile elements.

The 1D type tiling is a regular linear tiling with uniform tile sizes.
The 2D type tiling is a 2 dimensional cartesian tiling with uniform tile sizes,
that can be stored in two different orders, rowmajor and columnmajor.

With the tiling API, you can create and destroy a tiling. 
You can also perform some operations over a tiling. 
You can create and destroy an iterator, access the indexing, size of tiles or
their tiling dimensions.

Tiling High Level API
---------------------

.. doxygengroup:: aml_tiling

Implementations
---------------

There are so far two implementations for the AML tiling, in 1D and in 2D:

.. toctree::
   
   tiling_resize_api
   tiling_pad_api
