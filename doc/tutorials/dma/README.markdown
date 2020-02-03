# AML DMA Tutorial

## Prerequisite

Prior to start this tutorial you will need to install AML.
See AML (Readme)[../../README.markdown] for installation instructions.
If you did not install it to a standard location, make sure that
environment variables:
* LD_LIBRARY_PATH points to a directory where `libaml.so` is present
* C_INCLUDE_PATH points to a directory where `aml.h` and `aml/` are present.

## What is AML DMA ?

DMA is an acronym for Direct Memory Access.
DMA is a hardware accelerated method for moving data across
memory regions without the intervention of a compute unit.

AML DMA is a building block for moving data as well.
Data is moved in between two
(areas)[https://argo-aml.readthedocs.io/en/latest/pages/areas.html], i.e between
two virtual memory ranges represented by a pointer.
Data is moved from one
(layout)[https://argo-aml.readthedocs.io/en/latest/pages/layout.html]
to another, i.e data can be reorganized during the DMA request.
Data can also be moved asynchronously by a DMA engine, e.g pthreads on a CPU and
cuda streams on cuda accelerators.

The API for using AML DMA is broke down into two levels.
* The high-level API in the (main header)[../../include/aml.h] provides
generic functions that can be applied on all DMAs.
It also describes the general structure of a DMA for implementers.
* Implementations specific methods, constructors and static DMAs declarations
stand in the second level of headers (aml/dma/*.h)[../../include/aml/dma].

## Examples of AML DMA use cases:

* (Prefetching)[https://doi.org/10.1109/MCHPC49590.2019.00015]:
Writing an efficient matrix multiplication routine requires many architecture
specific optimizations such as vectorization, and cache blocking.
On hardware with software managed side memory cache (e.g MCDRAM on Intel
Knights Landing) manual prefetch of blocks in the side cache helps improving
performance on large problem sizes. AML DMA can help making the code
more compact by abstracting asynchronous memory movements. Moreover,
AML is also able to benefit from the prefetch time to reorganize data
such that matrix multiplication on the prefetched blocks will be vectorized.
See linked (publication)[https://doi.org/10.1109/MCHPC49590.2019.00015] for more
details.

* Replication:
(Some applications)[https://github.com/ANL-CESAR/XSBench] will have a
memory access pattern such that all threads will access one data in a read-only
and latency-bound fashion. On NUMA computing systems, accessing distant memories
will imply a penalty that translates in a penalty for application execution time.
In such a scenario (application + NUMA), replicating data on memories in order
to avoid NUMA penalties can result in significant performance improvements.
AML DMA is the building block to go when implementing simple interface for
performing data replication.
(This work has been accepted for publication to PHYSOR 2020 conference).

## How to use AML DMA ?

In order to perform a DMA request, you will need to set up a DMA, i.e
the engine that perform requests, then perform the request.
The request happens in between two layouts, therefore you will have
to setup layouts has well.

## Example

* Setting up the dma 
```
// DMA declaration.
struct aml_dma *dma;

// DMA initialization.
// See <aml/dma/linux-par.h> for function documentation.
// First argument is a pointer to the dma to initialize.
// Second argument is a number of pre-allocated DMA requests slots.
// Last arguments are not used here. They allow to set optimized internal
// functions for processing DMA requests.
aml_dma_linux_par_create(&dma, 128, NULL, NULL);
```

* Setting up data and layouts for simple copy.

Source data for copy is a simple 1D array of 8 elements.
```
double src[8] = {1, 2, 3, 4, 5, 6, 7, 8};
// src will have only one dimension of 8 elements.
size_t src_dims[1] = {8};
```
It translates into a dense layout as follow:
```
struct aml_layout *src_layout;
aml_layout_dense_create(&src_layout, src, AML_LAYOUT_ORDER_COLUMN_MAJOR, sizeof(*src), 1, src_dims, NULL, NULL);
```
Now we can define the destination of the copy in a similar way.
```
// The destination data for the move.
double dst[8] = {0, 0, 0, 0, 0, 0, 0, 0};
// dst will have only one dimension of 8 elements as well.
size_t dst_dims[1] = {8};

// The layout of destination data is here.
struct aml_layout *dst_layout;
aml_layout_dense_create(&dst_layout, dst, AML_LAYOUT_ORDER_COLUMN_MAJOR, sizeof(*dst), 1, dst_dims, NULL, NULL);
```
* Scheduling and waiting the DMA request.

We are all set to perform the copy.
We have created a DMA engine and described our source and destination data.
Let's do it with the high level API:
```
// Handle to the dma request we are about to issue.
struct aml_dma_request *request;
aml_dma_async_copy_custom(dma, &request, dst_layout, src_layout, NULL, NULL);

```
Now the dma request is on flight.
When we are ready to access data in dst, we can wait for it.
```
aml_dma_wait(dma, &request);
```

(Working code snippet)[./0_example.c]

## Exercises

### 1. Reduction

Let `a` a strided vector where contiguous elements are separated by a blank.
Let `b` a strided vector where contiguous elements are separated by 2 blanks.
Let `ddot` a function operating on two continuous vectors to perform a dot
product.
The goal is to transform `a` into `continuous_a` and `b` into `continuous_b`
in order to perform the dot product.

ddot definition is given here:
```
double ddot(const double *x, const double *y, const size_t n)
{
	double result = 0.0;
	for (i = 0; i < n; i++)
			result += x[i] * y[i];
	return result;
}
```

A possible value for `a` and its layout is given here:
```
double a[8] = {0.534, 6.3424, 65.4543, 4.543e12, 0.0, 1.0, 9.132e2, 23.657};
size_t a_dims[1] = {4}; // a has only 4 elements.
size_t a_stride[1] = {2}; // elements are strided by two.

struct aml_layout *a_layout;
aml_layout_dense_create(&a_layout, a, AML_LAYOUT_ORDER_COLUMN_MAJOR, sizeof(*a), 1, a_dims, a_stride, NULL);
```												

A possible value for `b` and its layout is given here:
```
double b[12] = {1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, };
size_t b_dims[1] = {4}; // b has 4 elements as well.
size_t b_stride[1] = {3}; // b elements are strided by three.

struct aml_layout *b_layout;
aml_layout_dense_create(&b_layout, b, AML_LAYOUT_ORDER_COLUMN_MAJOR, sizeof(*b), 1, b_dims, b_stride, NULL);
```
