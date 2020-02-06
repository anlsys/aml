Tilings
=======

What is an AML tiling ?
-----------------------

In AML, a tiling is a representation of the decomposition of data structures.
It identifies a way in which a layout can be split into layouts of smaller
sizes. 

As such, the main function of a tiling is to provide an index into
subcomponents of a layout.

As for the layouts, both the C and Fortran indexing orders are available for
the tilings, with similar names: `AML_TILING_ORDER_C` and
`AML_TILING_ORDER_FORTRAN`.

Creating an AML tiling
~~~~~~~~~~~~~~~~~~~~~~

First, you need to have the right headers.

.. code-block:: c

   #include <aml.h>
   #include <aml/layout/dense.h>
   #include "aml/tiling/resize.h"

Let's take the example of a two-dimensional matrix, with `x` rows and `y`
columns. We need to have already allocated some memory space for our matrix,
for instance with a AML area.

.. code-block:: c
								
   double mat[x][y];
   double *mat;
   struct aml_area *area = &aml_area_linux;
   mat = (double *)aml_area_mmap(area, sizeof(double) * x * y, NULL);

We then need to declare and create a layout in any order. 
Let's do both orders:

.. code-block:: c

   struct aml_layout *layout_c, *layout_f;
   size_t dims_col[2] = { x, y };

   aml_layout_dense_create(&layout_c, mat, AML_LAYOUT_ORDER_C, sizeof(double), 2, dims, NULL, NULL)); 

   aml_layout_dense_create(&layout_f, mat, AML_LAYOUT_ORDER_FORTRAN, sizeof(double), 2, dims, NULL, NULL));

After that, we can finally create a tiling associated to each layout, one in
`AML_TILING_ORDER_C`, the other in `AML_TILING_ORDER_FORTRAN`:

.. code-block:: c

   struct aml_tiling *tiling_c, *tiling_f;
   size_t tile_x = x/3, tile_y = y/3;

   aml_tiling_resize_create(&tiling_c, AML_TILING_ORDER_C, layout_c, 2, (size_t[]){tile_x, tile_y}));

   aml_tiling_resize_create(&tiling_f, AML_TILING_ORDER_FORTRAN, layout_f, 2, (size_t[]){tile_x, tile_y}));

We have just created two tilings on our two layouts, each with two dimensions, 
`{tile_x, tile_y}` for both tilings.
	
Getting the order of an AML tiling 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can see the order of a tiling by using the following function from the AML
Tiling API:

.. code-block:: c

   aml_tiling_order(tiling_c));
   aml_tiling_order(tiling_f));

The order of the first tiling is `AML_TILING_ORDER_C`, which in AML
is represented by the value 0, and the order of the second tiling is
`AML_TILING_ORDER_FORTRAN`, and the above function would return 1.

Destroying an AML tiling 
~~~~~~~~~~~~~~~~~~~~~~~~

In the same way you need to free the memory allocated for an array, you need to
destroy your tiling when you're done using it. 

.. code-block:: c

   aml_tiling_resize_destroy(&tiling_c);
   aml_tiling_resize_destroy(&tiling_f);


Generic operations on an AML tiling
-----------------------------------

Several operations on an AML tiling are defined in the AML Tiling generic API.
Let's assume here that we have successful created a tiling called `tiling` in
this part. 

We can get the number of dimensions of this tiling:

.. code-block:: c

   size_t ndims = aml_tiling_ndims(tiling);

In the previous examples, the number of dimensions of the tilings would be 2.

Once you've got the number of dimensions of the tiling, you can get the size of
each dimension in an array:

.. code-block:: c

   size_t dims[ndims];
   int err = aml_tiling_dims(tiling, dims);

This function will return a non-zero integer if there is an error, and 0 if
everything is fine.
In our previous tilings, the dimensions returned would be `{3, 3}` for
each tiling, because we cut our layouts into 3 in each dimension.

We can also get the number of tiles inside the tiling:

.. code-block:: c

   size_t ntiles = aml_tiling_ntiles(tiling);

This would have returned 9 for our previous tilings. 

The dimensions of each tile in the tiling can be obtained in a array:

.. code-block:: c

   size_t tile_dims[ndims];
   aml_tiling_tile_dims(tiling, tile_dims);

The resulting array would be `{tile_x, tile_y}` for both `tiling_c` and `tiling_f`.

Accessing a tile and elements of a tile
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can access any tile of the tiling by using its id in the tiling.

This will give you a tile, which is in fact a smaller layout than the one the
tiling is based on.

Once you have this layout, you can access each element of the tile with its
coordinates within this layout, of dimensions `tile_dims`.

Here is an example of going through each tile with the function
`aml_tiling_index_byid`, then going through each dimension of the tile, and
setting each element to 1.0 with the function `aml_layout_deref`:

.. code-block:: c

   size_t coords[ndims];
   double *a;

   for (size_t i = 0; i < ntiles; i++) {
      struct aml_layout *ltile;
      ltile = aml_tiling_index_byid(tiling, i);

      for (size_t j = 0; j < tile_dims[0]; j++) {
         for (size_t k = 0; k < tile_dims[1]; k++) {
            coords[0] = j;
            coords[1] = k; 
            a = aml_layout_deref(ltile, coords);
            *a = 1.0;
         }
      }
   }


Exercise
--------

Let's look at an example of when you could use tiles. 

Let `a` be a matrix of doubles of size `m*k`.
Let `b` be a matrix of doubles of size `k*n`.
We want to get the matrix `c` of doubles of size `m*n` which is the result of the matrix
multiplication of `a` and `b`.
We assume that `m`, `n` and `k` can be divided by 3.

We want to use the tilings to perform a blocked matrix multiplication.

In order to do so, you will need to first allocate the memory for your
matrices and initialize them, including `c` to all zeros. 

Then you will need to create a layout for each of them, each with two
dimensions, of the corresponding sizes for each matrix.
You can all create them with the same order.

Then, since all the dimensions of the matrices can be divided by 3, you will
need to create a tiling for each layout, with the dimensions of each tile being
3 times smaller than corresponding matrix dimensions.

After that you will be able to access each tile and do a blocked matrix
multiplication on each tile. 

Solution
~~~~~~~~

.. container:: toggle

   .. container:: header

      **Click Here to Show/Hide Code**

   .. literalinclude:: 1_dgemm.c
      :language: c

You can find this solution in *doc/tutorials/tiling*.								 

