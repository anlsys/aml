Area Linux Implementation 
=========================

This is the Linux implementation of AML areas.

This building block relies on the libnuma implementation and the Linux
mmap() / munmap() to provide mmap() / munmap() on NUMA host processor memory. 
New areas may be created to allocate a specific subset of memories.
This building block also includes a static declaration of a default initialized
area that can be used out-of-the-box with the abstract area API.

.. codeblock:: c
        #include <aml/area/linux.h

Example
-------
Using built-in feature of linux areas:
We allocate data accessible by several processes with the same address, spread
across all CPU memories (using linux interleave policy)

.. codeblock:: c
  // include ..

  struct aml_area* area;
  aml_area_linux_create(&area, AML_AREA_LINUX_MMAP_FLAG_SHARED, NULL,
                        AML_AREA_LINUX_BINDING_FLAG_INTERLEAVE);

  // When work is done with this area, free resources associated with it
  aml_area_linux_destroy(&area);

Integrating new feature in a new area implementation with some linux features:
You need an area feature not integrated in AML, but you want to work with AML
features around areas.
You can extend the features of linux area and reimplement a custom
implementation of mmap and munmap functions with
additional fields.

.. codeblock:: c
  // include ..

  // declaration of data field used in generic areas
  struct aml_area_data {
     // uses features of linux areas
     struct aml_area_linux_data linux_data;
     // implements additional features
     void* my_data;
  };

  // create your struct my_area_data with custom linux settings
  struct aml_area_data {
     .linux_data = {
         .nodeset = NULL,
         .binding_flags = AML_AREA_LINUX_BINDING_FLAG_INTERLEAVE,
         .mmap_flags = AML_AREA_LINUX_FLAG_SHARED,
     },
     .my_data = whatever_floats_your_boat,
  } my_area_data;
 
  // implements mmap using linux area features and custom features
  void* my_mmap(const struct aml_area_data* data, void* ptr, size_t size){
      program_data = aml_area_linux_mmap(data->linux_data, ptr, size);
      aml_area_linux_mbind(data->linux_data, program_data, size);
      // additional work we wnat to do on top of area linux work
      whatever_shark(data->my_data, program_data, size);
      return program_data;
  }
  // same for munmap
  int* my_munmap(cont struct aml_area_data* data, void* ptr, size_t size);

  // builds your custom area
  struct aml_area_ops {
     .mmap = my_mmap,
     .munmap = my_munmap,
  } my_area_ops;

  struct aml_area {
     .ops = my_area_ops,
     .data = my_area_data,
  } my_area;
  
  void* program_data = aml_area_mmap(&my_area, NULL, size);


And now you can call the generic API on your area.

Area Linux API
==============

.. doxygengroup:: aml_area_linux
