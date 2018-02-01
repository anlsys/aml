#include <assert.h>
#include <aml.h>
#include <jemalloc/jemalloc-aml.h>
#include <pthread.h>
#include <stdio.h>
#include <sys/mman.h>

/*******************************************************************************
 * Arena registry:
 * jemalloc extent hooks only receive the arena id when called, so we have to
 * maintain a registry of all arena allocated.
 ******************************************************************************/

/* MALLCTL_ARENAS_ALL is a reserved index and the last valid one */
#define AML_ARENA_MAX (MALLCTL_ARENAS_ALL-1)

struct aml_arena_jemalloc_global_data {
	struct aml_area *registry[AML_ARENA_MAX];
	struct aml_area *current;
	pthread_mutex_t lock;
};

static struct aml_arena_jemalloc_global_data aml_arena_jemalloc_global = {
	{NULL},
	NULL,
	PTHREAD_MUTEX_INITIALIZER,
};

static struct aml_area *aml_arena_registry_get(
				struct aml_arena_jemalloc_global_data *g,
				unsigned int arenaid)
{
	assert(g != NULL);
	assert(arenaid < AML_ARENA_MAX);
	struct aml_area *ret = g->registry[arenaid];
	if(ret == NULL)
		return g->current;
	else
		return ret;
}

/*******************************************************************************
 * Extent hooks:
 * jemalloc has "extent hooks" to create special arenas where the actual virtual
 * address space management is user-controlled. We use these to control memory
 * binding policies from our areas.
 ******************************************************************************/

/* when jemalloc asks for alignments that are bigger than PAGE_SIZE, the regular
 * mmap will not work, and we need to do extra work. */
static void* aml_arena_extra_align_alloc(struct aml_area *area, void *new_addr,
					 size_t size, size_t alignment)
{
	size_t big_size = size + alignment;
	void *addr;

	addr = aml_area_mmap(area, new_addr, big_size);
	if(addr == MAP_FAILED)
		return NULL;

	uintptr_t iaddr = (uintptr_t)addr;
	uintptr_t aligned_addr = (iaddr + alignment) & (alignment-1);

	size_t front_len = aligned_addr - iaddr;
	if(front_len > 0)
		munmap(addr, front_len);

	uintptr_t back = aligned_addr + size;
	size_t back_len = (iaddr + big_size) - (aligned_addr + size);
	if(back_len > 0)
		munmap((void *)back, back_len);

	return (void *)aligned_addr;
}

static void* aml_arena_extent_alloc(extent_hooks_t *extent_hooks,
					     void *new_addr, size_t size,
					     size_t alignment, bool *zero,
					     bool *commit, unsigned int arenaid)
{
	void *addr;
	struct aml_area *area =
		aml_arena_registry_get(&aml_arena_jemalloc_global, arenaid);

	if(!aml_area_available(area))
		return NULL;

	addr = aml_area_mmap(area, new_addr, size);
	if(addr == MAP_FAILED)
		return NULL;

	if(new_addr != NULL && addr != new_addr) {
		/* not mmaped in the right place */
		munmap(addr, size);
		return NULL;
	}

	if((uintptr_t)addr & (alignment-1)) {
		munmap(addr, size);
		addr = aml_arena_extra_align_alloc(area, new_addr, size,
						   alignment);
		if(addr == NULL)
			return addr;
	}

	*zero = true;
	*commit = true;
	return addr;
}

static bool aml_arena_extent_dalloc(extent_hooks_t *extent_hooks,
					     void *addr, size_t size,
					     bool committed,
					     unsigned arena_ind)
{
	return false;
}

static void aml_arena_extent_destroy(extent_hooks_t *extent_hooks,
					      void *addr, size_t size,
					      bool committed, unsigned arena_ind)
{
}

static bool aml_arena_extent_commit(extent_hooks_t *extent_hooks,
					     void *addr, size_t size,
					     size_t offset, size_t length,
					     unsigned arena_ind)
{
	return false;
}

static bool aml_arena_extent_decommit(extent_hooks_t *extent_hooks,
					       void *addr, size_t size,
					       size_t offset, size_t length,
					       unsigned arena_ind)
{
	return false;
}

static bool aml_arena_extent_purge(extent_hooks_t *extent_hooks,
					    void *addr, size_t size,
					    size_t offset, size_t length,
					    unsigned arena_ind)
{
	return false;
}

static bool aml_arena_extent_split(extent_hooks_t *extent_hooks,
					    void *addr, size_t size,
					    size_t size_a, size_t size_b,
					    bool committed, unsigned arena_ind)
{
	return false;
}

static bool aml_arena_extent_merge(extent_hooks_t *extent_hooks,
					    void *addr_a, size_t size_a,
					    void *addr_b, size_t size_b,
					    bool committed, unsigned arena_ind)
{
	return false;
}

static extent_hooks_t aml_arena_extent_hooks = {
	.alloc = aml_arena_extent_alloc,
	.dalloc = aml_arena_extent_dalloc,
	.commit = aml_arena_extent_commit,
	.decommit = aml_arena_extent_decommit,
	.purge_lazy = aml_arena_extent_purge,
	.split = aml_arena_extent_split,
	.merge = aml_arena_extent_merge
};

/*******************************************************************************
 * Core Arena Behavior:
 * Tunable by changing initialization flags
 ******************************************************************************/


int aml_arena_jemalloc_flags(int flags)
{
	int ret = 0;
	if(flags & AML_ARENA_FLAG_ZERO)
		ret |= MALLOCX_ZERO;
	return ret;
}

/* TODO: make the function idempotent */
int aml_arena_jemalloc_create(struct aml_arena_data *a, struct aml_area *area)
{
	int err;
	unsigned int newidx;
	struct aml_arena_jemalloc_data *arena =
		(struct aml_arena_jemalloc_data*) a;
	extent_hooks_t *hooks = &aml_arena_extent_hooks;
	size_t unsigned_size = sizeof(unsigned int);

	/* only one create at a time */
	pthread_mutex_lock(&aml_arena_jemalloc_global.lock);
	aml_arena_jemalloc_global.current = area;

	/* the locking above is required because this creation will end up
	 * calling the extent hooks before we have a change of registering the
	 * area.
	 */
	err = jemk_mallctl("arenas.create", &newidx, &unsigned_size, &hooks,
			   sizeof(hooks));
	if(err)
		goto exit;

	arena->uid = newidx;
	arena->flags |= MALLOCX_ARENA(newidx);
	aml_arena_jemalloc_global.registry[newidx] = area;
exit:
	aml_arena_jemalloc_global.current = NULL;
	pthread_mutex_unlock(&aml_arena_jemalloc_global.lock);
	return err;
}

/* TODO: make the function idempotent */
int aml_arena_jemalloc_purge(struct aml_arena_data *a)
{
	struct aml_arena_jemalloc_data *arena =
		(struct aml_arena_jemalloc_data*) a;
	char cmd[64];
	pthread_mutex_lock(&aml_arena_jemalloc_global.lock);

	snprintf(cmd, sizeof(cmd), "arena.%u.purge", arena->uid);
	jemk_mallctl(cmd, NULL, NULL, NULL, 0);
	aml_arena_jemalloc_global.registry[arena->uid] = NULL;

	pthread_mutex_unlock(&aml_arena_jemalloc_global.lock);
	return 0;
}

void *aml_arena_jemalloc_mallocx(struct aml_arena_data *a, size_t sz,
			       int extraflags)
{
	struct aml_arena_jemalloc_data *arena =
		(struct aml_arena_jemalloc_data*) a;
	int flags = arena->flags | aml_arena_jemalloc_flags(extraflags);
	return jemk_mallocx(sz, flags);
}

void *aml_arena_jemalloc_reallocx(struct aml_arena_data *a, void *ptr,
				  size_t sz, int extraflags)
{
	struct aml_arena_jemalloc_data *arena =
		(struct aml_arena_jemalloc_data*) a;
	int flags = arena->flags | aml_arena_jemalloc_flags(extraflags);
	return jemk_rallocx(ptr, sz, flags);
}

void aml_arena_jemalloc_dallocx(struct aml_arena_data *a, void *ptr,
				int extraflags)
{
	struct aml_arena_jemalloc_data *arena =
		(struct aml_arena_jemalloc_data*) a;
	int flags = arena->flags | aml_arena_jemalloc_flags(extraflags);
	jemk_dallocx(ptr, flags);
}

struct aml_arena_ops aml_arena_jemalloc_ops = {
	aml_arena_jemalloc_create,
	aml_arena_jemalloc_purge,
	aml_arena_jemalloc_mallocx,
	aml_arena_jemalloc_dallocx,
	aml_arena_jemalloc_reallocx,
};

/*******************************************************************************
 * Custom initializers:
 * To create the data template for arenas.
 ******************************************************************************/

int aml_arena_jemalloc_regular_init(struct aml_arena_jemalloc_data *data)
{
	assert(data != NULL);
	data->flags = 0;
	return 0;
}

int aml_arena_jemalloc_regular_destroy(struct aml_arena_jemalloc_data *data)
{
	assert(data != NULL);
	return 0;
}

int aml_arena_jemalloc_aligned_init(struct aml_arena_jemalloc_data *data,
				    size_t align)
{
	assert(data != NULL);
	data->flags = MALLOCX_ALIGN(align);
	return 0;
}

int aml_arena_jemalloc_align_destroy(struct aml_arena_jemalloc_data *data)
{
	assert(data != NULL);
	return 0;
}

int aml_arena_jemalloc_generic_init(struct aml_arena_jemalloc_data *data,
				    struct aml_arena_jemalloc_data *template)
{
	assert(data != NULL);
	assert(template != NULL);
	data->flags = template->flags;
	return 0;
}

int aml_arena_jemalloc_generic_destroy(struct aml_arena_jemalloc_data *data)
{
	assert(data != NULL);
	return 0;
}
