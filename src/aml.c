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

#include "allocator.h"

const char *tmpfs = "/tmp";
#define min(a,b) ((a) < (b)? (a) : (b))

int aml_init(int *argc, char **argv[])
{
	return 0;
}

int aml_finalize(void)
{
	return 0;
}

int aml_node_init(struct aml_node *node, struct bitmask *mask, size_t maxsize)
{
	char *template, zero[4096];
	size_t pos;
	ssize_t count;
	int fd;
	int mode;
	unsigned long oldmask[NUMA_NUM_NODES];
	assert(node != NULL);

	/* new, temporary file, to hold data */
	template = calloc(1024, sizeof(char));
	snprintf(template, 1024, "%s/%u.XXXXXX", tmpfs, getpid());
	fd = mkstemp(template);
	assert(fd != -1);

	/* as weird as it sounds, using mempolicy here forces the
	 * future writes to end up in the right memory node.
	 * Only necessary on first write to a page.
	 * We retrieve the current policy first to restore it later
	 */
	assert(!get_mempolicy(&mode, oldmask, NUMA_NUM_NODES, 0, 0));
	assert(!set_mempolicy(MPOL_BIND, mask->maskp, mask->size));

	/* write zeros all over to pull its pages in memory*/
	for(pos = 0; pos < maxsize; pos += count)
		if((count = write(fd, zero, min(maxsize - pos, 4096))) <= 0)
			break;

	/* init internal allocator */
	void *m = mmap(NULL, maxsize, PROT_READ|PROT_WRITE, MAP_PRIVATE,
		       fd, 0);
	assert(m != MAP_FAILED);
	aml_allocator_init(m, maxsize);
	munmap(m, maxsize);

	/* restore the original mempolicy */
	assert(!set_mempolicy(mode, oldmask, NUMA_NUM_NODES));

	node->path = template;
	node->fd = fd;
	node->maxsize = maxsize;
	return 0;
	return 0;
}

int aml_node_destroy(struct aml_node *node)
{
	assert(node != NULL);
	close(node->fd);
	unlink(node->path);
	return 0;
}

int aml_malloc(struct aml_alloc *a, size_t memsize, size_t blocksize,
	       struct aml_node *node)
{
	assert(a != NULL);
	assert(memsize % blocksize == 0);
	/* find one good initial pointer:
	 * the system will give us a start pointer so that the entire alloc can
	 * fit in memory.
	 */
	void *m = mmap(NULL, memsize, PROT_READ|PROT_WRITE, MAP_PRIVATE,
		       node->fd, 0);
	assert(m != MAP_FAILED);

	/* as long as nothing else is doing mmaps in our back, we can munmap
	 * and reuse the pointer immediately.
	 */
	munmap(m, memsize);
	a->start = m;
	a->memsize = memsize;
	a->blocksize = blocksize;
	a->numblocks = memsize/blocksize;
	a->nodemap = calloc(a->numblocks, sizeof(*a->nodemap));
	for(unsigned long i = 0; i < a->numblocks; i++)
	{
		a->nodemap[i] = NULL;
		aml_pull_sync(a, i, node);
	}
	return 0;
}

int aml_free(struct aml_alloc *a)
{
	assert(a != NULL);
	return munmap(a->start, a->memsize);
}

int aml_pull_sync(struct aml_alloc *a, unsigned long block,
		  struct aml_node *node)
{
	int flags = MAP_PRIVATE|MAP_FIXED;
	int prot = PROT_READ|PROT_WRITE;
	size_t offset;
	void *begin, *ret;
	assert(a != NULL);
	assert(block < a->numblocks);
	if(a->nodemap[block] != node)
	{
		offset = block*a->blocksize;
		begin = (void*)((unsigned long)a->start + offset);
		ret = mmap(begin, a->blocksize, prot, flags, node->fd, offset);
		assert(ret != MAP_FAILED && ret == begin);
		a->nodemap[block] = node;
	}
	return 0;
}

int aml_push_sync(struct aml_alloc *a, unsigned long block,
		  struct aml_node *node)
{
	return aml_pull_sync(a, block, node);
}
