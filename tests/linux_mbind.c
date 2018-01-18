#include <aml.h>
#include <assert.h>
#include <numa.h>
#include <numaif.h>
#include <stdio.h>
#include <sys/mman.h>

#define PAGE_SIZE 4096
#define ALLOC_SIZE (1<<20)
#define ALLOC_PAGES (ALLOC_SIZE/PAGE_SIZE)

/* some of that should probably end up in the lib itself */
#define __NBITS (8*sizeof(unsigned long))
#define __ELT(i) ((i) / __NBITS)
#define __MASK(i) ((unsigned long) 1 << ((i) % __NBITS))
#define BIT(mask, i) (mask[__ELT(i)] & __MASK(i))

/* validate that each page is on an authorized node */
void checkpages(void *ptr, unsigned long *mask)
{
	int i;
	void *pages[ALLOC_PAGES];
	int status[ALLOC_PAGES];
	for(i = 0; i < ALLOC_PAGES; i++)
		pages[i] = (void *)((char *)ptr + PAGE_SIZE);
	move_pages(0, ALLOC_PAGES, pages, NULL, status, 0);

	for(i = 0; i < ALLOC_PAGES; i++)
		assert(BIT(mask, status[i]));
}

void doit(struct aml_area_linux_mbind_data *config,
	  struct aml_area_linux_mbind_ops *ops,
	  int policy, unsigned long *mask)
{
	int err, mode;
	void *ptr;
	unsigned long rmask[AML_NODEMASK_SZ];

	err = ops->pre_bind(config);
	assert(err == 0);

	/* MAP_POPULATE is necessary to ensure our bindings are enforced in case
	 * mempolicy is used internally.
	 *
	 * Our bindings should resist those issues.
	 */
	ptr = mmap(NULL, ALLOC_SIZE, PROT_READ|PROT_WRITE,
		   MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
	assert(ptr != MAP_FAILED);

	err = ops->post_bind(config, ptr, ALLOC_SIZE);
	assert(err == 0);

	/* Retrieve the current policy for that alloc and compare.
	 * get_mempolicy does not return the right policy if it was set using
	 * set_mempolicy binding methods. As a result, we only check the mode
	 * and mask if a mode different form MPOL_DEFAULT is returned.
	 */
	get_mempolicy(&mode, rmask, AML_MAX_NUMA_NODES, ptr, MPOL_F_ADDR);
	if(mode != MPOL_DEFAULT) {
		assert(mode == policy);
		assert(!memcmp(rmask, mask, AML_NODEMASK_BYTES));
	}
	checkpages(ptr, mask);
	munmap(ptr, ALLOC_SIZE);
}

#define ARRAY_SIZE(x) (sizeof(x)/sizeof(x[0]))
struct aml_area_linux_mbind_ops *tocheck[] = {
	&aml_area_linux_mbind_regular_ops,
	&aml_area_linux_mbind_mempolicy_ops,
};

int main(int argc, char *argv[])
{
	struct aml_area_linux_mbind_data config;
	unsigned long nodemask[AML_NODEMASK_SZ];
	struct bitmask *allowed;
	int mode;

	/* library initialization */
	aml_init(&argc, &argv);

	/* retrieve the current nodemask:
	 * while in theory we can retrieve that info for get_mempolicy, the
	 * default binding policy returns an empty nodemask, so it doesn't
	 * really help us. We use the numa library directly instead.*/
	allowed = numa_get_mems_allowed();
	memcpy(nodemask, allowed->maskp, AML_NODEMASK_BYTES);

	/* use MPOL_BIND for checks, and make sure init worked. */
	aml_area_linux_mbind_init(&config, MPOL_BIND, nodemask);
	assert(config.policy == MPOL_BIND);
	assert(config.nodemask[0] == nodemask[0]);

	for(int i = 0; i < ARRAY_SIZE(tocheck); i++)
		doit(&config, tocheck[i], MPOL_BIND, nodemask);

	aml_area_linux_mbind_destroy(&config);
	aml_finalize();
	return 0;
}
