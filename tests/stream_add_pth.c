#include <aml.h>
#include <assert.h>
#include <errno.h>
#include <omp.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define ITER 10
#define MEMSIZE (1UL<<20)
#define CHUNKING 4

struct aml_area slow, fast;
struct aml_dma dma;

int kernel(unsigned long *a, unsigned long *b, unsigned long *c, size_t n)
{
	size_t i;
	printf("%p = %p + %p [%zi]\n",c,a,b,n);
	for(i = 0; i < n; i++)
		c[i] = a[i] + b[i];
	return 0;
}

struct cinfo {
	unsigned long *tab;
	pthread_t tid;
	size_t size;
};

void *th_copy(void *arg)
{
	struct cinfo *ci = arg;
	aml_dma_move(&dma, &fast, &slow, ci->tab, ci->size*sizeof(unsigned long));
	return arg;
}

struct winfo {
	unsigned long *a, *b, *c;
	pthread_t *ca, *cb;
	pthread_t tid;
	size_t size;
};

void *th_work(void *arg)
{
	struct winfo *wi = arg;
	pthread_join(*(wi->ca), NULL);
	pthread_join(*(wi->cb), NULL);

	kernel(wi->a, wi->b, wi->c, wi->size);
	return arg;
}
int main(int argc, char *argv[])
{
	assert(argc == 1);
	aml_init(&argc, &argv);

	/* we want to back our array on the slow node and use the fast node as
	 * a faster buffer.
	 */
	assert(!aml_area_from_nodestring(&slow, AML_AREA_TYPE_REGULAR, "0"));
	assert(!aml_area_from_nodestring(&fast, AML_AREA_TYPE_REGULAR, "0"));
	struct aml_dma dma;
	assert(!aml_dma_init(&dma, 0));

	void *a, *b, *c;

	/* describe the allocation */
	size_t chunk_msz, esz;
	int numthreads, copythreads;

	/* use openmp env to figure out how many threads we want
	 * (we actually use 3x as much)
	 */
	#pragma omp parallel
	{
		numthreads = omp_get_num_threads();
		chunk_msz = MEMSIZE/(numthreads*CHUNKING);
		esz = chunk_msz/sizeof(unsigned long);
	}
	a = aml_area_malloc(&slow, MEMSIZE);
	b = aml_area_malloc(&slow, MEMSIZE);
	c = aml_area_malloc(&fast, MEMSIZE);
	assert(a != NULL && b != NULL && c != NULL);

	/* create virtually accessible address range, backed by slow memory */
	unsigned long *wa = (unsigned long*)a;
	unsigned long *wb = (unsigned long*)b;
	unsigned long *wc = (unsigned long*)c;
	unsigned long esize = MEMSIZE/sizeof(unsigned long);
	for(unsigned long i = 0; i < esize; i++) {
		wa[i] = i;
		wb[i] = esize - i;
		wc[i] = 0;
	}

	/* run kernel */
	struct cinfo *cas = calloc(numthreads, sizeof(struct cinfo));
	struct cinfo *cbs = calloc(numthreads, sizeof(struct cinfo));
	struct winfo *wis = calloc(numthreads, sizeof(struct winfo));
	for(unsigned long i = 0; i < CHUNKING; i++) {
		for(unsigned long j = 0; j < numthreads; j++) {
			cas[j].tab = &wa[i*CHUNKING +j];
			cas[j].size = esize;
			cbs[j].tab = &wb[i*CHUNKING +j];
			cbs[j].size = esize;
			wis[j].a = &wa[i*CHUNKING +j];
			wis[j].b = &wb[i*CHUNKING +j];
			wis[j].c = &wc[i*CHUNKING +j];
			wis[j].ca = &cas[j].tid;
			wis[j].cb = &cbs[j].tid;
			wis[j].size = esize;
			pthread_create(&cas[j].tid, NULL, &th_copy, (void*)&cas[j]);
			pthread_create(&cbs[j].tid, NULL, &th_copy, (void*)&cbs[j]);
			pthread_create(&wis[j].tid, NULL, &th_work, (void*)&wis[j]);
		}
		for(unsigned long j = 0; j < numthreads; j++) {
			pthread_join(wis[j].tid, NULL);
		}
	}
	free(cas);
	free(cbs);
	free(wis);

	/* validate */
	for(unsigned long i = 0; i < esize; i++) {
		assert(wc[i] == esize);
	}

	aml_area_free(&slow, a);
	aml_area_free(&slow, b);
	aml_area_free(&fast, c);
	aml_area_destroy(&slow);
	aml_area_destroy(&fast);
	aml_dma_destroy(&dma);
	aml_finalize();
	return 0;
}
