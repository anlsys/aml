DMAs
====

What is an AML DMA ?
--------------------

In computer science, DMA (Direct Memory Access) is a hardware-accelerated
method for moving data across memory regions without the intervention of a
compute unit.

Similarly, in AML a DMA is the building block used to move data. Data is
generally moved between two `areas <../../pages/areas.html>`_, 
between two virtual memory ranges represented by a pointer.

Data is moved from one `layout <../../pages/layout.html>`_ to another.
When performing a DMA operation, layout coordinates are walked element by
element in
post order and matched to translate source coordinates into destination
coordinates.

Depending on the DMA implementation, this operation can be
optimized or offloaded to a DMA accelerator.
Data can thus be moved asynchronously by the DMA engine, e.g., pthreads on a CPU
and CUDA streams on CUDA accelerators.

The API for using AML DMA is broken down into two levels.

- The `high-level API <../../pages/dmas.html>`_ provides generic functions that can be applied on all DMAs. It also describes the general structure of a DMA for implementers.
- Implementation-specific methods, constructors, and static DMAs declarations reside in the second level of headers `<aml/dma/\*.h> <https://github.com/anlsys/aml/tree/master/include/aml/dma>`_.

Examples of AML DMA Use Cases
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- `Prefetching <https://doi.org/10.1109/MCHPC49590.2019.00015>`_:
	Writing an efficient matrix multiplication routine requires many
        architecture-specific optimizations such as vectorization and cache
        blocking.
	On hardware with software-managed side memory cache (e.g., MCDRAM on Intel
	Knights Landing), manual prefetch of blocks in the side cache helps improve
	performance for large problem sizes. AML DMA can help make the code
	more compact by abstracting asynchronous memory movements. Moreover,
	AML is also able to benefit from the prefetch time to reorganize data
	such that matrix multiplication on the prefetched blocks will be vectorized.
	See linked `publication <https://doi.org/10.1109/MCHPC49590.2019.00015>`_
	for more details.

- Replication:
	`Some applications <https://github.com/ANL-CESAR/XSBench>`_
	will have a memory access pattern such that all threads will access same data
	in a read-only and latency-bound fashion. On NUMA computing systems, accessing
	distant memories will imply a penalty that results in an increased
	application execution time. In such a scenario (application + NUMA),
	replicating data on memories in order to avoid NUMA penalties can result in
	significant performance improvements. AML DMA is the building block to go when
	implementing simple interface for performing data replication.
	(This work has been accepted for publication to PHYSOR 2020 conference).

Usage
-----

First, include the good headers.

.. code-block:: c
  
  #include <aml.h>
  #include <aml/dma/linux-par.h> // one DMA implementation.
  #include <aml/layout/dense.h> // one layout implementation.

First header contains `DMA generic API <../../pages/dmas.html>`_ and AML utils.
`Second header <../../pages/dma_linux_par_api.html>`_ will help build a
DMA-performing data transfer in the background with pthreads.
`Third header <../../pages/layout_dense.html>`_ will help describe source and
destination data to transfer.

In order to perform a DMA request, you will need to set up a DMA, i.e.,
the engine that performs requests, then perform the request itself.

.. code-block:: c
								
  struct aml_dma *dma;
  aml_dma_linux_par_create(&dma, 128, NULL, NULL);

We created a DMA that has 128 slots available off-the-shelf to
handle asynchronous data transfer requests.

The requests happen in-between two layouts, therefore you will have
to set up layouts as well. Let's suppose we want to copy `src` to `dst`

.. code-block:: c
								
  double src[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  double dst[8] = {0, 0, 0, 0, 0, 0, 0, 0};

The simplest copy requires that both `src` and `dst` are one-dimensional
layouts of 8 elements.

.. code-block:: c
								
  size_t dims[1] = {8};

For a one-dimension layout, dimension order does not matter, so let's pick
`AML_LAYOUT_ORDER_COLUMN_MAJOR`. Now we can initialize layouts.

.. code-block:: c

  struct aml_layout *src_layout, *dst_layout;
  aml_layout_dense_create(&dst_layout, dst, AML_LAYOUT_ORDER_COLUMN_MAJOR, sizeof(*dst), 1, dims, NULL, NULL);
  aml_layout_dense_create(&src_layout, src, AML_LAYOUT_ORDER_COLUMN_MAJOR, sizeof(*src), 1, dims, NULL, NULL);

We have created a DMA engine and described our source and destination data.
We are all set to schedule a copy DMA request.

.. code-block:: c

  struct aml_dma_request *request;
  aml_dma_async_copy_custom(dma, &request, dst_layout, src_layout, NULL, NULL);

Now the DMA request is in-flight.
When we are ready to access data in dst, we can wait for it.
	
.. code-block:: c

  aml_dma_wait(dma, &request);

Exercise
--------

Let `a` be a strided vector where contiguous elements are separated by a blank.
Let `b` be a strided vector where contiguous elements are separated by 2 blanks.
Let `ddot` be a function operating on two continuous vectors to perform a dot
product.
The goal is to transform `a` into `continuous_a` and `b` into `continuous_b`
in order to perform the dot product.

ddot definition is given below:

.. code-block:: c

	double ddot(const double *x, const double *y, const size_t n)
	{
		double result = 0.0;
		for (i = 0; i < n; i++)
			result += x[i] * y[i];
		return result;
	}


A possible value for `a` and its layout is given here:

.. code-block:: c

  double a[8] = {0.534, 6.3424, 65.4543, 4.543e12, 0.0, 1.0, 9.132e2, 23.657};
  size_t a_dims[1] = {4}; // a has only 4 elements.
  size_t a_stride[1] = {2}; // elements are strided by two.

  struct aml_layout *a_layout;
  aml_layout_dense_create(&a_layout, a, AML_LAYOUT_ORDER_COLUMN_MAJOR, sizeof(*a), 1, a_dims, a_stride, NULL);

A possible value for `b` and its layout is given here:

.. code-block:: c

  double b[12] = {1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, };
  size_t b_dims[1] = {4}; // b has 4 elements as well.
  size_t b_stride[1] = {3}; // b elements are strided by three.

  struct aml_layout *b_layout;
  aml_layout_dense_create(&b_layout, b, AML_LAYOUT_ORDER_COLUMN_MAJOR, sizeof(*b), 1, b_dims, b_stride, NULL);	

Solution
~~~~~~~~

.. container:: toggle

   .. container:: header

      **Click Here to Show/Hide Code**

   .. literalinclude:: 1_reduction.c
      :language: c

You can find this solution in *doc/tutorials/dma/*.								 

