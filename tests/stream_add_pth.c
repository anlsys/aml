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

struct aml_node slow, fast;

int kernel(unsigned long *a, unsigned long *b, unsigned long *c, size_t n)
{
	size_t i;
	printf("%p = %p + %p [%zi]\n",c,a,b,n);
	for(i = 0; i < n; i++)
		c[i] = a[i] + b[i];
	return 0;
}

struct cinfo {
	struct aml_alloc *tab;
	pthread_t tid;
	unsigned long chunk;
};

void *th_copy(void *arg)
{
	struct cinfo *ci = arg;
	aml_block_move(ci->tab, ci->chunk, &fast);
	return arg;
}

struct winfo {
	struct aml_alloc *a, *b, *c;
	pthread_t *ca, *cb;
	pthread_t tid;
	unsigned long chunk;
};

void *th_work(void *arg)
{
	struct winfo *wi = arg;
	pthread_join(*(wi->ca), NULL);
	pthread_join(*(wi->cb), NULL);

	void *aa,*bb,*cc;
	size_t esize = aml_block_size(wi->c)/sizeof(unsigned long);
	
	aml_block_address(wi->a, wi->chunk, &aa);
	aml_block_address(wi->b, wi->chunk, &bb);
	aml_block_address(wi->c, wi->chunk, &cc);
	printf("%p[%lu]:%p\n",wi->a->start, wi->chunk, aa);
	printf("%p[%lu]:%p\n",wi->b->start, wi->chunk, bb);
	printf("%p[%lu]:%p\n",wi->c->start, wi->chunk, cc);
	kernel(aa, bb, cc, esize);
	return arg;
}
int main(int argc, char *argv[])
{
	assert(argc == 1);
	aml_init(&argc, &argv);

	/* we want to back our array on the slow node and use the fast node as
	 * a faster buffer.
	 */
	assert(!aml_node_init(&slow, 0));
	assert(!aml_node_init(&fast, 0));

	struct aml_alloc a,b,c;
	
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
	printf("th: %lu, mem: %zi, chunk: %zi\n",numthreads,MEMSIZE,chunk_msz);
	assert(!aml_malloc(&a, MEMSIZE, chunk_msz, &slow));
	assert(!aml_malloc(&b, MEMSIZE, chunk_msz, &slow));
	assert(!aml_malloc(&c, MEMSIZE, chunk_msz, &fast));

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
	struct cinfo *cas = calloc(numthreads, sizeof(struct cinfo));
	struct cinfo *cbs = calloc(numthreads, sizeof(struct cinfo));
	struct winfo *wis = calloc(numthreads, sizeof(struct winfo));
	for(unsigned long i = 0; i < CHUNKING; i++) {
		for(unsigned long j = 0; j < numthreads; j++) {
			cas[j].tab = &a;
			cas[j].chunk = i*CHUNKING + j;
			cbs[j].tab = &b;
			cbs[j].chunk = i*CHUNKING + j;
			wis[j].a = &a;
			wis[j].b = &b;
			wis[j].c = &c;
			wis[j].ca = &cas[j].tid;
			wis[j].cb = &cbs[j].tid;
			wis[j].chunk = i*CHUNKING + j;
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

	aml_free(&a);
	aml_free(&b);
	aml_free(&c);
	aml_node_destroy(&slow);
	aml_node_destroy(&fast);
	aml_finalize();
	return 0;
}
