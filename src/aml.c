#include <aml.h>
#include <assert.h>
#include <fcntl.h>
#include <numa.h>
#include <numaif.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>

#ifndef PAGE_SIZE
#define PAGE_SIZE 4096
#endif

#ifndef MAX_NUMNODES 64
#define MAX_NUMNODES 64
#endif

int aml_init(int *argc, char **argv[])
{
	return 0;
}

int aml_finalize(void)
{
	return 0;
}

int aml_node_init(struct aml_node *node, unsigned int nid)
{
	assert(node != NULL);
	assert(nid < MAX_NUMNODES);
	node->numaid = nid;
	node->mask = numa_bitmask_alloc(MAX_NUMNODES);
	numa_bitmask_setbit(node->mask, nid);
	return 0;
}

int aml_node_destroy(struct aml_node *node)
{
	assert(node != NULL);
	free(node->mask);
	return 0;
}

int aml_malloc(struct aml_alloc *a, size_t memsize, size_t blocksize,
	       struct aml_node *node)
{
	assert(a != NULL);
	assert(memsize % blocksize == 0);
	assert(blocksize % PAGE_SIZE == 0);
	/* TODO: convert to SICM */
	struct bitmask *oldbind = numa_get_membind();
	numa_set_membind(node->mask);
	void *m = mmap(NULL, memsize, PROT_READ|PROT_WRITE,
		       MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
	assert(m != MAP_FAILED);
	memset(m, 0, memsize);
	numa_set_membind(oldbind);

	/* start tracking blocks */
	a->start = m;
	a->memsize = memsize;
	a->blocksize = blocksize;
	a->numblocks = memsize/blocksize;
	a->nodemap = calloc(a->numblocks, sizeof(*a->nodemap));
	for(unsigned long i = 0; i < a->numblocks; i++)
		a->nodemap[i] = node;
	return 0;
}

int aml_free(struct aml_alloc *a)
{
	assert(a != NULL);
	free(a->nodemap);
	a->nodemap = NULL;
	return munmap(a->start, a->memsize);
}

int aml_block_address(struct aml_alloc *a, size_t block, void **ret)
{
	assert(a != NULL);
	assert(block < a->numblocks);
	*ret = (void*)((char*)a->start + block*a->blocksize);
	return 0;
}

int aml_block_move(struct aml_alloc *a, size_t block, struct aml_node *node)
{
	assert(a != NULL);
	assert(block < a->numblocks);
	if(a->nodemap[block] != node) {
		unsigned long count = a->blocksize/PAGE_SIZE;
		int *nodes = calloc(count, sizeof(*nodes));
		void **pages = calloc(count, sizeof(*pages));
		int *status = calloc(count, sizeof(*status));
		for(unsigned long i = 0; i < count; i++) {
			nodes[i] = node->numaid;
			pages[i] = (void*)((char*)a->start + i*PAGE_SIZE);
		}
		move_pages(0, count, pages, nodes, status, MPOL_MF_MOVE);
	}
	return 0;
}

int aml_block_copy(struct aml_alloc *src, size_t srcblock,
		   struct aml_alloc *dest, size_t destblock)
{
	return 0;
}
