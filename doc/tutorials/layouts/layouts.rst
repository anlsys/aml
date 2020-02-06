Layouts
=======

What is an AML layout ?
-----------------------

In AML, a layout is an abstraction over the indexing of a contiguous virtual
address space (buffer). In other words, it describes how data is stored in
memory and needs to be dereferenced. A layout does not necessarily own the
backing memory buffer, and multiple layouts can be used to provide different
views into the data.

Let's take the example of a 1 dimensional array numbered sequentially:

.. code-block:: c

  double array[size];
  for (size_t i = 0; i < size; i++)
        array[i] = i;

And let's look at how different 2d (matrix) layouts provide a view into this
array.

In Fortran, the elements of a matrix are stored by encasing the last dimension into
the previous one and so on. This means that the elements of the last dimension
will be contiguous in memory, i.e. the elements of one column will be stored
contiguously.

In C, it is the reverse, the elements of the first dimension are
contiguous in memory, which means for a 2d matrix that the elements of one row
would be contiguous in memory.

The layouts in AML allow you to choose the order in which the program iterates
over the dimensions. This order can be specified when creating a layout.

Creating an AML layout
~~~~~~~~~~~~~~~~~~~~~~

First, you need to have the right headers.

.. code-block:: c

   #include <aml.h>
   #include <aml/layout/dense.h>

You need to have already have allocated some memory space for your data, for
instance with a AML area.

.. code-block:: c

   double *array;
   struct aml_area *area = &aml_area_linux;
   array = (double *)aml_area_mmap(area, sizeof(double) * size, NULL);

Then you can declare and create a layout that should be iterated like in C
(last dimension moves the fastest). Let's
use `x` and `y` for the 2d dimensions (with `x * y = size`):

.. code-block:: c

   struct aml_layout *layout_c;
   size_t dims[2] = { x, y };
   aml_layout_dense_create(&layout_c, array, AML_LAYOUT_ORDER_C, sizeof(double), 2, dims, NULL, NULL);

We have just created a layout with elements of type double, with two
dimensions, the fist dimension containing `x` elements, and the second
dimension containing `y` elements.

In the same way, you can create your layout to be iterated like in Fortran
(first dimension moves the fastest).

.. code-block:: c

   struct aml_layout *layout_f;
   aml_layout_dense_create(&layout_f, array, AML_LAYOUT_ORDER_FORTRAN, sizeof(double), 2, dims, NULL, NULL);

We have just created a second layout with elements of type `double`, with two
dimensions, the fist dimension containing `x` elements, and the second
dimension containing `y` elements.

Note that the two layouts have been defined over the same memory location.

The two arguments set to NULL in the above creation function are meant for
layouts created on data not contiguous in memory, allowing to set strides and
pitches to describe exactly the storage situation.

Getting the order of an AML layout
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can see the order of a layout by using the following function from the AML
Layout API:

.. code-block:: c

   aml_layout_order(layout_c));
   aml_layout_order(layout_f));

The order of the first layout is `AML_LAYOUT_ORDER_C`, which in AML is
represented by the value 0, and the order of the second layout is
`AML_LAYOUT_ORDER_FORTRAN`, and the above function would return 1.

Destroying an AML layout 
~~~~~~~~~~~~~~~~~~~~~~~~

In the same way you need to free the memory allocated for an array, you need to
destroy your layout when you're done using it.

.. code-block:: c

   aml_layout_dense_destroy(&layout_c);
   aml_layout_dense_destroy(&layout_f);


Generic operations on an AML layout
-----------------------------------

Several operations on an AML layout are defined in the AML Layout generic API.
Let's assume here that we have successful created a layout called `layout` in
this part.

We can get the number of dimensions of this layout:

.. code-block:: c

   size_t ndims = aml_layout_ndims(layout);

In the previous examples, the number of dimensions of the layouts would be 2.

Once you've got the number of dimensions of the layout, you can get the size of
each dimension in an array:

.. code-block:: c

   size_t dims[ndims];
   int err = aml_layout_dims(layout, dims);

This function will return a non-zero integer if there is an error, and 0 if
everything is fine. In our previous layouts, the dimensions returned would be
`{x, y}` for both `layout_c`, and `layout_f`.

You can also get the size of one element of the layout:

.. code-block:: c

   size_t element_size = aml_layout_element_size(layout);

This would have returned the `sizeof(double)` for our previous layouts.

Accessing elements of a layout
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can access any element stored in the layout by using its coordinates in the
system indexed by the layout's dimensions `dims`.
Here is an example of going through each dimension of the layout, and
reading each element with the function `aml_layout_deref`.

.. code-block:: c

   size_t coords[ndims];
   double *a;

   for (size_t i = 0; i < dims[0]; i++) {
      for (size_t j = 0; j < dims[1]; j++) {
         coords[0] = i;
         coords[1] = j;
         a = aml_layout_deref(layout, coords);
         printf("%f ", *a);
      }
      printf("\n");
   }

You can find this code in *doc/tutorials/layouts/0_dense_layout.c*. Try to
predict what the output will be for each layout used. Experiment with different
layouts.

Changing the shape of a layout
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also change the shape of your layout, creating a new layout with a
different number of dimensions.

Let's take the previous layout `layout_c`, that had two dimensions `{x, y}`,
and create a new layout with three dimensions, basically splitting the first
dimension in two:

.. code-block:: c

   size_t new_dims[3] = { x/2, x/2, y };
   struct aml_layout *reshape_layout;
   aml_layout_reshape(layout_c, &reshape_layout, 3, new_dims));

The new layout is ordered in the same order as the previous layout, in this
case `AML_LAYOUT_ORDER_C`.

You can also want to mix up the order of the dimensions of your layout.
This cannot be done with the `reshape` function. You need to allocate a new
memory area, create another layout with the right dimensions, and copy the
elements from one layout to the other.

Let's say we have a three-dimensional layout `layout_3` and we want to run a
permutation on the dimensions and get a layout `new_layout` with the first
dimension of `layout_3` in third place, the second one in first and the last
one in second place.
After allocating the correct memory area and creating `new_layout` with the
right dimensions, we would use `aml_copy_layout_transform_generic`

.. code-block:: c

   struct aml_area *area = &aml_area_linux;
   array_1 = (double *)aml_area_mmap(area, sizeof(double) * size_0 * size_1 * size_2, NULL);
   array_2 = (double *)aml_area_mmap(area, sizeof(double) * size_0 * size_1 * size_2, NULL);

   struct aml_layout *layout_3, *new_layout;
   aml_layout_dense_create(&layout_3, array_1, AML_LAYOUT_ORDER_C, sizeof(size_t), 3, (size_t[]){size_0, size_1, size_2}, NULL, NULL));

   aml_layout_dense_create(&new_layout, array_2, AML_LAYOUT_ORDER_C, sizeof(size_t), 3, (size_t[]){size_1, size_2, size_0}, NULL, NULL));

   aml_copy_layout_transform_generic(new_layout, layout_3, (size_t[]){1, 2, 0}));


Exercise
--------

Let's look at an example of when you could use layouts.

Let `particle` be a data structure with several attributes:

.. code-block:: c

   struct particle {
      size_t id;
      size_t position_x;
      size_t position_y;
      double energy;
   };

Let `particles` be a two-dimensional array of particles.

We basically have an array of structures, which is hard to manipulate. We want
to use the AML layouts to transform this into a structure of arrays, that could allow
more performance on some platforms.

Based on the above layout, a straightforward layout on this array could be:

.. code-block:: c

   struct aml_layout *layout_part;
   aml_layout_dense_create(&layout_part, particles, AML_LAYOUT_ORDER_C, sizeof(struct particle), 2, (size_t[]){size_1, size_2}, NULL, NULL));

The issue with this layout is that it does not give us a fine enough control on
the attributes of the particles. We need to be able to index each field of the
structure independently. This can be done by adding an extra dimension (because
all fields have the same storage size here):

.. code-block:: c

   struct aml_layout *layout_elements;

   aml_layout_dense_create(&layout_elements, particles, AML_LAYOUT_ORDER_C, sizeof(size_t), 3, (size_t[]){4, size_1, size_2}, NULL, NULL));

Now we can create another layout, with a similar granularity, but with the
dimensions flipped so that we have one array for each attribute of all the
particles.  Then we need to copy the elements of the first layout in the second
layout in the right order, using the function
`aml_copy_layout_transform_generic`.

Solution
~~~~~~~~

.. container:: toggle

   .. container:: header

      **Click Here to Show/Hide Code**

   .. literalinclude:: 2_aos_soa.c
      :language: c

You can find this solution in *doc/tutorials/layouts*.

