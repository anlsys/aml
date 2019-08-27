Layout: Description of Data Organization
========================================

A layout describes how contiguous elements of a flat memory address space are
organized into a multidimensional array of fixed-size elements.
The abstraction provides functions to build layouts, access elements, reshape a 
layout, or subset a layout.

A layout is characterized by:
* A pointer to the data it describes
* A set of dimensions on which data spans.
* A stride in between elements of a dimension.
* A pitch indicating the space between contiguous elements of a dimension.

The figure below describes a 2D layout with a sub-layout (obtained with 
aml_layout_slice()) operation.
The sub-layout has a stride of 1 element along the second dimension.
The slice has an offset of 1 element along the same dimension, and its pitch is
the pitch of the original layout.
Calling aml_layout_deref() on this sublayout with appropriate coordinates will
return a pointer to elements noted (coor_x, coord_y).

.. image:: img/layout.png 
   :width=400px
"2D layout with a 2D slice."

Access to specific elements of a layout can be done with the aml_layout_deref()
function.
Access to an element is always done relatively to the dimension order set by at
creation time.
However, internally, the library will store dimensions from the last dimension
to the first dimension such that elements along the first dimension are 
contiguous in memory. 
This order is defined with the value AML_LAYOUT_ORDER_FORTRAN. 
Therefore, AML provides access to elements without the overhead of user order
choice through function suffixed with "native".

The layout abstraction also provides a function to reshape data with a different
set of dimensions.
A reshaped layout will access the same data but with different coordinates as
pictured in the figure below.

.. image:: img/reshape.png 
   :width=700px
"2D layout turned into a 3D layout."
 
Example
-------

Let's look at a problem where layouts can be quite useful: matrix
multiplication, with DGEMM.
Let's say you want to multiple matrix A (size [m, k]) with matrix B 
(size [k, n]) to get matrix C (size [m, n]).

The first step is implementing an efficient micro-kernel. 
The micro-kernel update a block of C of size [mr, nr] noted C_r using a block of
A of size [mr, kb] noted A_r, and a block of B of size [kb, nr] noted B_r.
A_r is stored in column major order while C_r and B_r are stored in row major
order.

The medium kernel works using blocks of intermediate size. 
The medium kernel updates a block of C of size [kb, n] noted C_b using a block
of A of size [mb, kb] noted A_b, and a block of B of size [kb, n] noted B_b.
A_b is stored as mb/mr consecutive blocks of size [mr, kb] (A_r) in column major
order while C_b is stored as (mb/mr)*(n/nr) blocks of size [mr, nr] (C_r) in row
major order and B_b is stored as n/nr blocks of size [kb, nr] (B_r) in row major
order.

The large kernel uses matrices of any size.
Let's say we consider the matrices already transformed.
The original matrices are C of size [m, n], A of size [m, k] and B of size 
[k, n].
The layout used here are: C is stored as m/mb blocks of C_b, A is stored as
(k/kb) * (m/mb) blocks of A_b and B is stored as k/kb blocks of B_b.


High level API
--------------

.. doxygengroup:: aml_layout

Implementations
---------------

   .. toctree::

      layout_dense
      layout_native
      
