#include <assert.h>
#include <errno.h>
#include <omp.h>
#include <aml.h>
#include <stdlib.h>

#define ITER 10
#define MEMSIZE (1UL<<26)
#define PHASES 20
#define CHUNKING 4

int kernel(unsigned long *tab, size_t elems)
{
	size_t i;
	unsigned int r;
	for(r = 0; r < ITER; r++) {
		for(i = 1; i < elems -1; i++)
			tab[i] = tab[i-1] + tab[i] + tab[i+1];
	}
	return 0;
}

int main(int argc, char *argv[])
{
	assert(argc == 1);
	aml_init(&argc, &argv);

	/* we want to back our array on the slow node and use the fast node as
	 * a faster buffer.
	 */
	struct aml_node slow, fast;
	struct bitmask *mask = numa_parse_nodestring_all("0");
	assert(!aml_node_init(&slow, mask, MEMSIZE));
	assert(!aml_node_init(&fast, mask, MEMSIZE));

	/* we are only dealing with one contiguous array */
	struct aml_alloc alloc;

	/* describe the allocation */
	size_t chunk_msz, chunk_esz;
	int numthreads;

	#pragma omp parallel
	{
		numthreads = omp_get_num_threads();
		chunk_msz = MEMSIZE/(numthreads*CHUNKING);
		chunk_esz = chunk_msz/sizeof(unsigned long);
	}
	assert(!aml_malloc(&alloc, MEMSIZE, chunk_msz, &slow));

	/* create virtually accessible address range, backed by slow memory */
	unsigned long *wa = (unsigned long*)alloc.start;
	for(unsigned long i = 0; i < MEMSIZE/sizeof(unsigned long); i++) {
		wa[i] = i;
	}

	/* run kernel */
	#pragma omp parallel
	#pragma omp single nowait
	{
		for(unsigned long phase = 0; phase < PHASES; phase++) {
			for(unsigned long i = 0; i < numthreads*CHUNKING; i++) {
				#pragma omp task depend(inout: wa[i*chunk_esz:chunk_esz])
				assert(!aml_pull_sync(&alloc, i, &fast));
				#pragma omp task depend(inout: wa[i*chunk_esz:chunk_esz])
				kernel(&wa[i*chunk_esz], chunk_esz);
				#pragma omp task depend(inout: wa[i*chunk_esz:chunk_esz])
				assert(!aml_push_sync(&alloc, i, &slow));
			}
		}
	}
	aml_free(&alloc);
	aml_node_destroy(&slow);
	aml_node_destroy(&fast);
	aml_finalize();
	return 0;
}
