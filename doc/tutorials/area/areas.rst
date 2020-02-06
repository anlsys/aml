Areas
=====

AML areas represent places where data can be read and write.
Typically an area is an object from which you can query and free memory.
AML lends these actions mmap() and munmap() after linux system call to
map physical memory in the virtual address space.

On NUMA processor, mapping of data can be as complex as interleaving chunks
on several physical memories and making them appear as contiguous is user land.
On accelerator, physical memory of the device can be mapped with virtual
address range that is mapped on the host physical memory.
AML builds areas on top of libnuma and cuda to reach these levels of
customization while keeping the memory queries/free as simple as a function call.

As compute nodes get bigger, building relevant areas to manage locality
is likely to improve performance of memory-bound applications.
AML areas provide functions to query memory from specific places materialized as
areas. Available area implementations dictate the way such
places can be arranged and their properties. AML areas is a
low-level/high-overhead abstraction and is not intended to be optimized for
fragmented small allocations. Instead it intends to be a basic block for
implementing better allocators.

The API of AML Area is broke down into two levels.

- The `high-level API <../../pages/areas.html>`_ provides generic functions that can be applied on all areas. It also describes the general structure of an area for implementers.
- Implementations specific methods, constructors and static areas declarations stand in the second level of headers `<aml/area/\*.h> <https://xgitlab.cels.anl.gov/argo/aml/tree/master/include/aml/area>`_.

Use Cases
-------------

- Building custom memory mapping policies: high bandwidth memory only, interleave custom block sizes.
- Abstracting allocators memory mapping.

Usage
-----

You need to include at least two headers.

.. code-block:: c
  
  #include <aml.h> // General high level API
  #include <aml/area/linux.h> // One area implementation.

From there, you can already query memory from the processor.
`Linux area <../../pages/area_linux_api.html>`_ implementation provides
a static declaration of a default area: `aml_area_linux`.

.. code-block:: c

  void *data = aml_area_mmap(&aml_area_linux, 4096, NULL);

Here we have allocated 4096 Bytes of data available in `data` field.
This data can later be freed as follow.

.. code-block:: c

  aml_area_munmap(aml_area_linux, data, 4096);

Linux Area
----------

If you are working on a NUMA processor, you eventually want more
control on your memory provider. For instance you might want your data
to be spread on all memories to balance the load. One way to achieve it
is to use interleave linux policy. This policy can be applied when
building a custom linux area.

.. code-block:: c

  struct aml_area *interleave_area;
  aml_area_linux_create(&interleave_area, NULL, AML_AREA_LINUX_POLICY_INTERLEAVE);

Now we have an "allocator" of interleaved data.

.. code-block:: c

  void *data = aml_area_mmap(interleave_area, 4096*8, NULL);

Here we have allocated 8*4096 Bytes of data across system memories.

CUDA Area
---------

If you compiled AML on a cuda capable node, you will be able to use
AML cuda implementations of its building blocks.
It is possible to allocate cuda devices memory with aml,
in a very similar way as with linux implementation.

.. code-block:: c

  #include <aml.h> // General high level API
  #include <aml/cuda/linux.h> // Cuda area implementation.
  void *data = aml_area_mmap(&aml_area_cuda, 4096, NULL);

The pointer obtained from this allocation is a device side pointer.
It can't be directly read and written from host processor.

Exercise: CUDA Mirror Allocation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As an exercise, dive into `<aml/cuda/linux.h>` header and create an area
that will hand out pointer that can be read and written both on host and
device side. Check that modifications on host side are mirrored on device side.

.. container:: toggle

   .. container:: header

      **Click Here to Show/Hide Code**

   .. literalinclude:: 0_aml_area_cuda.c
      :language: c

You can find this solution in *doc/tutorials/area*.

Implementing a Custom Area
--------------------------

You might want to use AML blocks with a different area behaviour that is not
part of AML. This is achievable by implementing the area building block to
match the desired behavior.
In short, all AML building blocks consist in attributes stored in `data` field
and methods stored in `ops` field. In the case of area, `struct aml_area_ops`
require that custom mmap, munmap, and fprintf fields are implemented.
Let's implement an empty area. This area will have no attributes, i.e data
is NULL and its operation will print a message.
We first implement area methods.

.. code-block:: c

  #include <aml.h> // General high level API

  void* _mmap(const struct aml_area_data *data, size_t size, struct aml_area_mmap_options *opts) {
    (void) data; (void) size; (void) opts; // ignore arguments
    printf("mmap called.\n");
    return NULL;
  }

  int _munmap(const struct aml_area_data *data, void *ptr, size_t size) {
    (void) data; (void) ptr; (void) size; // ignore arguments
    printf("munmap called.\n");
    return AML_SUCCESS;
  }

  int _fprintf(const struct aml_area_data *data, FILE *stream, const char *prefix) {
    (void) data; // ignore argument
    fprintf(stream, "%s: fprintf called.\n", prefix);
  }

Now we can declare the area methods and area itself.

.. code-block:: c

  // Area methods declaration
  struct aml_area_ops _ops = {
    .mmap = _mmap,
    .munmap = _munmap,
    .fprintf = _fprintf,
  };

  // Area declaration
  struct aml_area _area = {
    .data = NULL,
    .ops = &_ops,
  };

Let's try it out:

.. code-block:: c

  aml_area_mmap(&_area, 4096, NULL);
  // "mmap called."
  aml_area_minmap(&_area, NULL, 4096);
  // "munmap called."
	
Exercise: interleaving in blocks of 2 pages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With the use of mbind() function from libnuma, implement an area
that will interleave blocks of 2 pages on the system memories.
For instance, let a system with 4 NUMA nodes and a buffer of
16 pages. Pages have to be allocated as follow:

.. code-block:: c

  page: [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 ]
  NUMA: [ 0, 0, 1, 1, 2, 2, 3, 3, 0, 0,  1,  1,  2,  2,  3,  3 ]

You can retrieve the size of a page the following way:

.. code-block:: c

  #include <unistd.h>
  sysconf(_SC_PAGESIZE);

You can test if your data is interleaved as requested with below code.
	
.. container:: toggle

   .. container:: header

      **Click Here to Show/Hide Code**

   .. code-block:: c
											
				// Function to get last NUMA node id on which data is allocated.
				static int get_node(void *data)
				{
					long err;
					int policy;
					unsigned long maxnode = sizeof(unsigned long) * 8;
					unsigned long nmask = 0;
					int node = -1;
				
					err = get_mempolicy(&policy, &nmask, maxnode, data, MPOL_F_ADDR);
					if (err == -1) {
						perror("get_mempolicy");
						exit(1);
					}
				
					while (nmask != 0) {
						node++;
						nmask = nmask >> 1;
					}
				
					return node;
				}
				
				// Check if data of size `size` is interleaved on all nodes,
				// by chunk of size `page_size`.
				static int is_interleaved(void *data, const size_t size, const size_t page_size)
				{
					intptr_t start;
					int node, next, num_nodes = 0;
				
					start = ((intptr_t) data) << page_size >> page_size;
				
					node = get_node((void *)start);
					// more than one node in policy.
					if (node < 0)
						return 0;
				
					for (intptr_t page = start + page_size;
					     (size_t) (page - start) < size; page += page_size) {
						next = get_node((void *)page);
				
						// more than one node in page policy.
						if (next < 0)
							return 0;
						// not round-robin
						if (next != (node + 1) && next != 0)
							return 0;
						// cycling on different number of nodes
						if (num_nodes != 0 && next >= num_nodes)
							return 0;
						// cycling on different number of nodes
						if (num_nodes != 0 && next == 0 && num_nodes != node)
							return 0;
						// set num_nodes
						if (num_nodes == 0 && next == 0)
							num_nodes = node;
						node = next;
					}
				
					return 1;
				}

Solution:

.. container:: toggle

   .. container:: header

      **Click Here to Show/Hide Code**

   .. literalinclude:: 1_custom_interleave_area.c
      :language: c

