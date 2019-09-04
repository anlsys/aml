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

Let's look at a problem where layouts can be quite useful: DGEMM in multiple
levels.
Let's say you want to multiply matrix A (size [m, k]) with matrix B 
(size [k, n]) to get matrix C (size [m, n]).

The naive matrix multiplication algorithm should look something like:

.. code:: c
    for (i = 0; i < m; i++){
        for (j = 0; j < n; j++){
            cij = C[i*n + j];
            for (l = 0; l < k; l++)
                cij += A[i*n + l] * B[l*n + j];
            C[i*n + j] = cij;
        }
    }

Unfortunately this algorithm does not have a great runtime complexity...

We can then have 3 nested loops running on blocks of the matrices. 
With several sizes of memory, we want to lverage the power of using blocks of
different sizes. 
Let's take an algorithm with three levels of granularity.


The first level is focused on fitting our blocks in the smallest cache. 
We compute a block of C of size [mr, nr] noted C_r using a block of
A of size [mr, kb] noted A_r, and a block of B of size [kb, nr] noted B_r.
A_r is stored in column major order while C_r and B_r are stored in row major
order, allowing us to read A_r row by row, and go with B_r and C_r column by
column.

.. code:: c
    for (i = 0; i < m_r; i++){
        for (j = 0; j < n_r; j++){
            for (l = 0; l < k_b; l++)
                C_r[i][j] += A_r[i][l] + B_r[l][j];
        }
    }

These are our smallest blocks. 
The implementation at this level is simply doing the multiplication at a level
where is fast enough.
B_r blocks need to be transposed before they can be accessed column by column.

The second level is when the matrices are so big that you need a second
caching.
We then use blocks of intermediate size. 
We compute a block of C of size [mb, n] noted C_b using a block
of A of size [mb, kb] noted A_b, and a block of B of size [kb, n] noted B_b.
To be efficient, A_b is stored as mb/mr consecutive blocks of size [mr, kb]
(A_r) in column major order while C_b is stored as (mb/mr)*(n/nr) blocks of
size [mr, nr] (C_r) in row major order and B_b is stored as n/nr blocks of size
[kb, nr] (B_r) in row major order.

This means we need to have Ab laid out as a 3-dimensional array mr x kb x (mb/mr),
B as nr x kb x (n/nr), C with 4 dimensions as nr x mr x (mb/mr) x (n/nr).

The last level uses the actual matrices, of any size.
The original matrices are C of size [m, n], A of size [m, k] and B of size 
[k, n].
The layout used here are: C is stored as m/mb blocks of C_b, A is stored as
(k/kb) * (m/mb) blocks of A_b and B is stored as k/kb blocks of B_b.

This means we need to rework A to be laid out in 5 dimensions as
mr x kb x mb/mr x m/mb x k/kb,
B in 4 dimensions as nr x kb x n/nr x k/kb,
C in 5 dimensions as nr x mr x mb/mr x n/nr x m/mb

High level API
--------------

.. doxygengroup:: aml_layout

Implementations
---------------

   .. toctree::

      layout_dense
      layout_native
      
