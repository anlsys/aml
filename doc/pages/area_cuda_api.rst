Area Cuda Implementation API
=================================
Cuda Implementation of Areas.

.. codeblock:: c
        #include <aml/area/cuda.h>
 
Cuda implementation of AML areas.
This building block relies on Cuda implementation of
malloc/free to provide mmap/munmap on device memory.
Additional documentation of cuda runtime API can be found here:
https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html

AML cuda areas may be created to allocate current or specific cuda devices.
Also allocations can be private to a single device or shared across devices.
Finally allocations can be backed by host memory allocation.

.. doxygengroup:: aml_area_cuda
