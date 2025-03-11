Tilings: Decomposing Data
====================================

Tiling is a representation of the decomposition of data structures.
It identifies ways in which a layout can be split into layouts of smaller
sizes.
As such the main function of a tiling is to provide an index into subcomponents
of a layout.
Implementations focus on the ability to provide sublayouts of different sizes
at the corners, and linearization of the index range.


Tiling High Level API
---------------------

.. doxygengroup:: aml_tiling

Implementations
---------------

There are so far two implementations for the AML tiling, in 1D and in 2D:

.. toctree::
   
   tiling_resize_api
   tiling_pad_api
