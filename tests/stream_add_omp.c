#include <assert.h>
#include <errno.h>
#include <omp.h>
#include <aml.h>
#include <stdlib.h>

#define ITER 10
#define MEMSIZE (1UL<<26)
#define PHASES 20
#define CHUNKING 4

int kernel(unsigned long *a, unsigned long *b, unsigned long *c, size_t n)
{
	size_t i;
	for(i = 0; i < n; i++)
		c[i] = a[i] + b[i];
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
	assert(!aml_node_init(&slow, mask, MEMSIZE*3));
	assert(!aml_node_init(&fast, mask, MEMSIZE*3));

	/* we are only dealing with one contiguous array */
	struct aml_alloc a,b,c;

	/* describe the allocation */
	size_t chunk_msz, esz;
	int numthreads;

	#pragma omp parallel
	{
		numthreads = omp_get_num_threads();
		chunk_msz = MEMSIZE/(numthreads*CHUNKING);
		esz = chunk_msz/sizeof(unsigned long);
	}
	assert(!aml_malloc(&a, MEMSIZE, chunk_msz, &slow));
	assert(!aml_malloc(&b, MEMSIZE, chunk_msz, &slow));
	assert(!aml_malloc(&c, MEMSIZE, chunk_msz, &slow));

	/* create virtually accessible address range, backed by slow memory */
	unsigned long *wa = (unsigned long*)a.start;
	unsigned long *wb = (unsigned long*)b.start;
	unsigned long *wc = (unsigned long*)c.start;
	unsigned long esize = MEMSIZE/sizeof(unsigned long);
	for(unsigned long i = 0; i < esize; i++) {
		wa[i] = i;
		wb[i] = esize - i;
		wc[i] = 0;
	}

	/* run kernel */
	#pragma omp parallel
	#pragma omp single nowait
	{
		for(unsigned long i = 0; i < numthreads*CHUNKING; i++) {
			#pragma omp task depend(inout: wa[i*esz:esz])
			assert(!aml_pull_sync(&a, i, &fast));
			#pragma omp task depend(inout: wb[i*esz:esz])
			assert(!aml_pull_sync(&b, i, &fast));
			#pragma omp task depend(inout: wc[i*esz:esz])
			assert(!aml_pull_sync(&c, i, &fast));
			#pragma omp task depend(in: wa[i*esz:esz], wb[i*esz:esz]) depend(out: wc[i*esz:esz])
			kernel(&wa[i*esz], &wb[i*esz], &wc[i*esz], esz);
			#pragma omp task depend(inout: wa[i*esz:esz])
			assert(!aml_push_sync(&a, i, &slow));
			#pragma omp task depend(inout: wb[i*esz:esz])
			assert(!aml_push_sync(&b, i, &slow));
			#pragma omp task depend(inout: wc[i*esz:esz])
			assert(!aml_push_sync(&c, i, &slow));
		}
	}

	/* validate */
	for(unsigned long i = 0; i < esize; i++) {
		assert(wc[i] == esize);
	}

	aml_free(&a);
	aml_free(&b);
	aml_free(&c);
	aml_node_destroy(&slow);
	aml_node_destroy(&fast);
	aml_finalize();
	return 0;
}
