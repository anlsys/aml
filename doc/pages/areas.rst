Areas: Addressable Physical Memories
====================================

AML areas represent places where data can be stored.
In shared memory systems, locality is a major concern for performance.
Being able to query memory from specific places is of major interest to achieve
this goal.
AML areas provide low-level mmap() / munmap() functions to query memory from
specific places materialized as areas. 
Available area implementations dictate the way such places can be arranged and
their properties.
 
.. image:: img/area.png 
   :width=700px
"Illustration of areas on a complex system."

An AML area is an implementation of memory operations for several type of
devices through a consistent abstraction.
This abstraction is meant to be implemented for several kind of devices, i.e.
the same function calls allocate different kinds of devices depending on the
area implementation provided.

With the high level API, you can:

* Use an area to allocate space for your data
* Release the data in this area

Example
-------

Let's look how these operations can be done in a C program.

.. codeblock:: c
  #include <aml.h>
  #include <aml/area/linux.h>

  int main(){

      void* data = aml_area_mmap(&aml_area_linux, s); 
      do_work(data);
      aml_area_munmap(data, s);
  }

We start by importing the AML interface, as well as the area implementation we
want to use.

We then proceed to allocate space for the data of size s using the default from
the AML Linux implementation.
The data will be only visible by this process and bound to the CPU with the
default linux allocation policy.

Finally, when the work is done with data, we free it.


Area API
--------

It is important to notice that the functions provided through the Area API are
low-level functions and are not optimized for performance as allocators are.

.. doxygengroup:: aml_area


Implementations
---------------
Aware users may create or modify implementation by assembling appropriate
operations in an aml_area_ops structure.

The linux implementation is go to for using simple areas on NUMA CPUs with
linux operating system. 

There is an ongoing work on hwloc, CUDA and OpenCL areas.

Let's look at an example of a dynamic creation of a linux area identical to the
static default aml_area_linux:

.. codeblock:: c
  #include <aml.h>
  #include <aml/area/linux.h>

  int main(){
      struct aml_area* area;
      aml_area_linux_create(&area, AML_AREA_LINUX_MMAP_FLAG_PRIVATE, NULL,
                        AML_AREA_LINUX_BINDING_FLAG_DEFAULT);
      do_work(area);
      aml_area_linux_destroy(&area);
  }

.. toctree::
   
   area_linux_api
   area_cuda_api
   area_opencl_api	 
   area_ze_api	 
