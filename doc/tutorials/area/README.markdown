# AML interleave tutorials

This tutorial aims at learn how to use and implement AML areas.
Through this exercise, you will:
* Use existing AML area to allocate data on several memories
in an interleave fashion.
* Implement a custom AML area that you will further
use to allocate data in an interleave fashion.
* Use existing AML area to allocate data on CUDA device.

# Tutorials

## [0_isInterleaved](./0_isInterleaved.c)

This tutorial set up test functions to check weather
a buffer is allocated in a round robin fashion.

The check looks if the memory policy associated with
the buffer is MPOL_INTERLEAVE, or if every pages in
the buffer are allocated on the machine nodes in a
round-robin fashion.

We further allocate buffers in such manners and
check that the tests are reporting the good result.

The result of this exercise is then put in the file
[tutorials.h](./tutorials.h) to check the next exercises.

### Exercise

```
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>		// sysconf()
#include <sys/mman.h>	// mmap()
#include <numa.h>     // numa_get_max_node()
#include <numaif.h>		// get_mempolicy()

/**
 * Check whether linux inerleave binding policy
 * is set on this range of address.
 **/
int is_interleaved_policy(void *data)
{
	// Exercise
}

/**
 * Walk all pages and check if they are bound in a round-robin
 * fashion.
 **/
int is_interleaved_bind(void *data, const size_t size)
{
		// Exercise
}

/**
 * data is interleaved if the interleave binding policy is set
 * or if pages are bound in a round-robin fashion.
 **/
int is_interleaved(void *data, const size_t size)
{
	return is_interleaved_policy(data) || is_interleaved_bind(data, size);
}

void *mmap_interleave_policy(const size_t size)
{
	const unsigned long NUMA_NODES[8] = { 1, 1, 1, 1, 1, 1, 1, 1 };
	void *out = mmap(NULL, size,
			 PROT_READ | PROT_WRITE,
			 MAP_PRIVATE | MAP_ANONYMOUS,
			 0, 0);

	if (out == NULL) {
		perror("mmap");
		exit(1);
	}

	if (mbind(out, size, MPOL_INTERLEAVE, NUMA_NODES,
		  sizeof(NUMA_NODES) / sizeof(*NUMA_NODES),
		  MPOL_MF_MOVE) == -1) {
		perror("mbind");
		munmap(out, size);
		exit(1);
	}

	return out;
}

void *mmap_interleave_bind(const size_t size)
{
	const int numnode = numa_max_node() == 0 ? 1 : numa_max_node();
	const unsigned long maxnode = sizeof(unsigned long) * 8;
	unsigned long NUMA_NODES = 1;
	int page_size = sysconf(_SC_PAGESIZE);
	intptr_t start;
	int node = 0;
	void *out;

	out = mmap(NULL, size, PROT_READ | PROT_WRITE,
		   MAP_PRIVATE | MAP_ANONYMOUS, 0, 0);

	if (out == NULL) {
		perror("mmap");
		exit(1);
	}

	start = ((intptr_t) out) << page_size >> page_size;
	for (intptr_t page = start; (size_t) (page - start) < size;
	     page += page_size) {

		NUMA_NODES = 1 << node;
		if (mbind((void *)page, page_size, MPOL_BIND, &NUMA_NODES,
			  maxnode, MPOL_MF_MOVE) == -1) {
			perror("mbind");
			munmap(out, size);
			exit(1);
		}
		node = (node + 1) % numnode;
	}

	return out;
}

int main(void)
{
	void *buf;
	const size_t size = (2 << 16);	// 16 pages

	// Check that linux bind interleave policy passes test.
	buf = mmap_interleave_policy(size);
	if (!is_interleaved(buf, size))
		return 1;
	printf("mmap_interleave_policy check works!\n");
	munmap(buf, size);

	// Check that binding pages in a round-robin fashion passes test.
	buf = mmap_interleave_bind(size);
	if (!is_interleaved(buf, size))
		return 1;
	printf("mmap_interleave_bind check works!\n");
	munmap(buf, size);

	return 0;
}
```

## [1_aml_area_linux](./1_aml_area_linux.c)

This tutorial will lead you to allocate a buffer with
aml area linux in an interleave fashion.
The result should be checked with the is_interleaved()
function implemented in previous exercise.

### Exercise

```
#include <stdlib.h>
#include <stdio.h>
#include <aml.h>
#include <aml/area/linux.h>
#include "tutorials.h"

/**
 * Find out how to fill the holes by looking into
 * header <aml/area/linux.h>
 **/
void test_default_area(const size_t size)
{
	void *buf;

	buf = aml_area_mmap(..., size, NULL);
	if (buf == NULL) {
		aml_perror("aml_area_linux");
		exit(1);
	}
	printf("Default linux area worked!\n");
	
	aml_area_munmap(..., buf, size);
}

/**
 * Instanciate an area that will map memory
 * on several NUMA nodes in an interleave fashion.
 * Then, Allocate a buffer with this area and
 * check if it is indeed interleaved.
 **/
void test_interleave_area(const size_t size)
{
	int err;
	void *buf;
	struct aml_area *interleave_area;

	// Create interleave area on all nodes.
	// ...
	
  // Map buffer in area.
	// ...

	// Check it is indeed interleaved
	if (!is_interleaved(buf, size))
		exit(1);
	printf("Interleave linux area worked and is interleaved.\n");

	// Cleanup
	// ...
}

int main(void)
{
	const size_t size = (2 << 16);	// 16 pages

	test_default_area(size);
	test_interleave_area(size);

	return 0;
}
```

## [2_custom_interleave_area](./2_custom_interleave_area.c)

This tutorial will learn you how to implement a custom
AML area to include into AML ecosystem. The custom area
must interleave data on NUMA nodes. The result must also
be checked with the is_interleaved() function from first
exercise. Look into (aml.h)[../../include/aml.h] to see
how area structure is shaped. Look into
(aml/area/linux.h)[../../include/aml/area/linux.h] to see
an example of area implementation.

### Exercise

```
#include <stdlib.h>
#include <stdio.h>
#include <aml.h>
#include "tutorials.h"

/**
 * Fill the structure that will populate the field
 * data from struct aml_area. This field contain
 * the information you will need during mmap call
 * to do the interleaving.
 **/
struct area_data {
			 // ...
};

/**
 * Implement your own mmap operator for aml_area_ops.
 **/
void *custom_mmap(const struct aml_area_data *data,
		  size_t size, struct aml_area_mmap_options *opts)
{
	(void)opts; // no need to use opts.
	// Cast generic data to your data structure.
	struct area_data *area = (struct area_data *)data;
	void *ret; // The pointer with memory to return.

	// ...

	return ret;
}

/**
 * Implement your own munmap operator for aml_area_ops.
 **/
int custom_munmap(const struct aml_area_data *data, void *ptr, size_t size)
{
 // ...
}

/**
 * Implement a constructor for your own area.
 **/
struct aml_area *custom_area_create()
{
	struct area_data *data;
	struct aml_area_ops *ops;
	struct aml_area *ret;

	// ...
	return ret;
}

void test_custom_area(const size_t size)
{
	void *buf;
	struct aml_area *interleave_area = custom_area_create();

	// Map buffer in area.
	buf = aml_area_mmap(interleave_area, size, NULL);
	if (buf == NULL) {
		aml_perror("aml_area_linux");
		exit(1);
	}
	// Check it is indeed interleaved
	if (!is_interleaved(buf, size))
		exit(1);
	printf("Custom area worked and is interleaved.\n");

	// Cleanup
	aml_area_munmap(interleave_area, buf, size);
	free(interleave_area);
}

int main(void)
{
	const size_t size = (2 << 16);	// 16 pages

	test_custom_area(size);
	return 0;
}
```

## [3_aml_area_cuda](./3_aml_area_cuda.c)

In this tutorial you will learn how to use aml
to allocate data on cuda devices and how to map
existing host data on cuda devices.

### Exercise

```
#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <aml.h>
#include <aml/area/cuda.h>

/**
 * Find out how to fill the holes by looking into
 * header <aml/area/cuda.h>
 **/
void test_default_area(const size_t size, void *host_buf)
{
	void *device_buf;

	// Map data on current device.
	device_buf = aml_area_mmap(..., size, NULL);
	if (device_buf == NULL) {
		aml_perror("aml_area_cuda");
		exit(1);
	}
	// Check we can perform a data transfert to mapped device memory.
	assert(cudaMemcpy(device_buf,
			  host_buf,
			  size, cudaMemcpyHostToDevice) == cudaSuccess);

	printf("Default cuda area worked!\n");

	// Cleanup
	aml_area_munmap(&..., device_buf, size);
}

/**
 * Create a cuda area that can map host memory to device memory.
 * Then map host_buf with device memory.
 * Finally set host_buf and check that device side memory,
 * has transparently been updated with the same values.
 **/
void test_custom_area(const size_t size, void *host_buf)
{
	int err;
	void *device_buf;
	struct aml_area *area;

	// Create an area that will map data on device 0, with host memory.
	// ...

	// Map host memory with device memory.
	// ...
	
	// Get device memory that is mapped with host memory.
	assert(cudaHostGetDevicePointer(&device_buf, host_buf, 0) ==
	       cudaSuccess);

	// Set data from host.
	// ...
	
	// Check that data on device has been set to the same value,
	// i.e mapping works.
	// ...

	// Cleanup
	aml_area_munmap(...);
	aml_area_cuda_destroy(...);
}

int main(void)
{
	// Skip tutorial if this is not supported.
	if (!aml_support_backends(AML_BACKEND_CUDA))
		return 77;

	const size_t size = (2 << 16);	// 16 pages
	void *host_buf = mmap(NULL, size,
			      PROT_READ | PROT_WRITE,
			      MAP_PRIVATE | MAP_ANONYMOUS,
			      0, 0);
	if (host_buf == NULL)
		return 1;

	test_default_area(size, host_buf);
	test_custom_area(size, host_buf);

	munmap(host_buf, size);
	return 0;
}
```
