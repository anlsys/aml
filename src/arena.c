#include <aml.h>
#include <assert.h>
#include <jemalloc/jemalloc.h>
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
static struct aml_area *aml_arena_registry[AML_ARENA_MAX];
static struct aml_area *current_init_area = NULL;
static pthread_mutex_t aml_arena_registry_lock = PTHREAD_MUTEX_INITIALIZER;


static struct aml_area *aml_arena_registry_get(unsigned int arenaid)
{
	assert(arenaid < AML_ARENA_MAX);
	struct aml_area *ret = aml_arena_registry[arenaid];
	if(ret == NULL)
		return current_init_area;
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

	addr = area->mmap(area, new_addr, big_size);
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
	struct aml_area *area = aml_arena_registry_get(arenaid);

	if(!area->available(area))
		return NULL;

	addr = area->mmap(area, new_addr, size);
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

int aml_arena_jemalloc_init(struct aml_arena *arena, struct aml_area *area)
{
	int err;
	unsigned int newidx;
	extent_hooks_t *hooks = &aml_arena_extent_hooks;
	size_t unsigned_size = sizeof(unsigned int);

	/* only one create at a time */
	pthread_mutex_lock(&aml_arena_registry_lock);
	current_init_area = area;

	/* while jemalloc provides a way to activate custom hooks directly on
	 * creation of an arena, it triggers the hooks too early. */
	err = jemk_mallctl("arenas.create", &newidx, &unsigned_size, &hooks,
			   sizeof(hooks));
	if(err)
		goto exit;

	arena->uid = newidx;
	aml_arena_registry[newidx] = area;
exit:
	current_init_area = NULL;
	pthread_mutex_unlock(&aml_arena_registry_lock);
	return err;
}

int aml_arena_jemalloc_destroy(struct aml_arena *arena)
{
	char cmd[64];
	pthread_mutex_lock(&aml_arena_registry_lock);
	
	snprintf(cmd, sizeof(cmd), "arena.%u.purge", arena->uid);
	jemk_mallctl(cmd, NULL, NULL, NULL, 0);
	aml_arena_registry[arena->uid] = NULL;
	
	pthread_mutex_unlock(&aml_arena_registry_lock);
	return 0;
}

void *aml_arena_jemalloc_malloc(struct aml_arena *arena, size_t sz)
{
	int flags = MALLOCX_ZERO | MALLOCX_ARENA(arena->uid);
	if(sz == 0)
		return NULL;
	return jemk_mallocx(sz, flags);
}

void aml_arena_jemalloc_free(struct aml_arena *arena, void *ptr)
{
	int flags = MALLOCX_ARENA(arena->uid);
	if(ptr == NULL)
		return;
	jemk_dallocx(ptr, flags);
}

void *aml_arena_jemalloc_calloc(struct aml_arena *arena, size_t nb, size_t sz)
{
	return aml_arena_jemalloc_malloc(arena, nb*sz);
}

void *aml_arena_jemalloc_realloc(struct aml_arena *arena, void *ptr, size_t sz)
{
	int flags = MALLOCX_ARENA(arena->uid);
	if(ptr == NULL)
		return aml_arena_jemalloc_malloc(arena, sz);
	if(sz == 0)
	{
		aml_arena_jemalloc_free(arena, ptr);
		return NULL;
	}	
	return jemk_rallocx(ptr, sz, flags);
}

void *aml_arena_jemalloc_acquire(struct aml_arena *arena, size_t sz)
{
	int flags = MALLOCX_ARENA(arena->uid);
	return jemk_mallocx(sz, flags);
}

void aml_arena_jemalloc_release(struct aml_arena *arena, void *ptr)
{
	aml_arena_jemalloc_free(arena, ptr);
}

struct aml_arena aml_arena_jemalloc = {
	0,
	aml_arena_jemalloc_init,
	aml_arena_jemalloc_destroy,
	aml_arena_jemalloc_malloc,
	aml_arena_jemalloc_free,
	aml_arena_jemalloc_calloc,
	aml_arena_jemalloc_realloc,
	aml_arena_jemalloc_acquire,
	aml_arena_jemalloc_release,
	NULL,
};


int aml_arena_init(struct aml_arena *arena, struct aml_arena *template,
		   struct aml_area *area)
{
	memcpy(arena, template, sizeof(*arena));
	return arena->init(arena, area);
}

int aml_arena_destroy(struct aml_arena *arena)
{
	return arena->destroy(arena);
}

