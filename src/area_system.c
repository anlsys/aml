#include <aml.h>
#include <assert.h>
#include <sys/mman.h>

#define MAX_NUMA_NODES 64
#define NODEMASK_SZ (MAX_NUMA_NODES/sizeof(unsigned long))

/*******************************************************************************
 * Arena mapper:
 * Can return a different arena depending on which threads ask for it.
 ******************************************************************************/

struct aml_area_system_arena_mapper {
	struct aml_arena *(*get_arena)(struct aml_area_system_mapper *);
	void *extra;
};

int aml_area_system_arena_mapper_single_init(
				struct aml_area_system_arena_mapper *mapper,
				struct aml_area *area)
{
	assert(mapper != NULL);
	struct aml_arena *myarena = malloc(sizeof(myarena));
	assert(myarena != NULL);
	mapper->extra = myarena;
	return aml_arena_init(myarena, &aml_arena_jemalloc, area);
}

int aml_area_system_arena_mapper_single_destroy(
				struct aml_area_system_arena_mapper *mapper)
{
	assert(mapper != NULL);
	assert(mapper->extra != NULL);
	return aml_arena_destroy(mapper->extra);
}

struct aml_arena * aml_area_system_single_get_arena(
				struct aml_area_system_arena_mapper *mapper)
{
	return mapper->extra;
}

struct aml_area_system_arena_mapper aml_area_system_arena_mapper_single = {
	aml_area_system_arena_mapper_single_get_arena,
	NULL
};

/*******************************************************************************
 * Binding policy:
 * Binds virtual memory according to a given policy
 ******************************************************************************/

struct aml_area_system_binder {
	int (*pre_bind)(struct aml_area_system_binder *);
	int (*post_bind)(struct aml_area_system_binder *, void *, size_t);
	unsigned long nodemask[NODEMASK_SZ];
	int policy;
};

int aml_area_system_binder_mbind_init(struct aml_area_system_binder *self,
				      unsigned long *nodemask,
				      int policy)
{
	assert(self != NULL);
	memcpy(self->nodemask, nodemask, NODEMASK_SZ);
	self->policy = policy;
	

int aml_area_system_binder_mbind_pre_bind(struct aml_area_system_binder *self)
{
	assert(self != NULL);
	return 0;
}

int aml_area_system_binder_mbind_post_bind(struct aml_area_system_binder *self,
					   void *ptr, size_t sz)
{
	assert(self != NULL);
	assert(self->nodemask != NULL);
	return mbind(ptr, sz, self->policy, self->nodemask, MAX_NUMA_NODES);
}

int aml_area_system_binder_mbind_destroy(struct aml_area_system_binder *self)
{
	assert(self != NULL);
	assert(self->nodemask != NULL);
	return 0;
}

struct aml_area_system_binder aml_area_system_binder_mbind_generic = {
	aml_area_system_binder_mbind_pre_bind,
	aml_area_system_binder_mbind_post_bind,
	{0},
	0,
};

int aml_area_system_binder_mempolicy_init(struct aml_area_system_binder *self,
					  unsigned long *nodemask,
					  int policy)
{
	assert(self != NULL);
	assert(nodemask != NULL);
	memcpy(self->nodemask, nodemask, NODEMASK_SZ);
	self->policy = policy;
}

int aml_area_system_binder_mempolicy_pre_bind(struct aml_area_system_binder *self)
{
	assert(self != NULL);
	/* save the old one, apply the new one */
	int policy;
	unsigned long nodemask[NODEMASK_SZ];
	get_mempolicy(&policy, nodemask, MAX_NUMA_NODES, NULL, 0);
	set_policy(self->policy, self->nodemask, MAX_NUMA_NODES);
	memcpy(self->nodemask, nodemask, NODEMASK_SZ);
	self->policy = policy;
	return 0;
}

int aml_area_system_binder_mempolicy_post_bind(struct aml_area_system_binder *self,
					   void *ptr, size_t sz)
{
	assert(self != NULL);
	/* save the old one, apply the new one */
	int policy;
	unsigned long nodemask[NODEMASK_SZ];
	get_mempolicy(&policy, nodemask, MAX_NUMA_NODES, NULL, 0);
	set_policy(self->policy, self->nodemask, MAX_NUMA_NODES);
	memcpy(self->nodemask, nodemask, NODEMASK_SZ);
	self->policy = policy;
}

int aml_area_system_binder_mbind_destroy(struct aml_area_system_binder *self)
{
	assert(self != NULL);
	return 0;
}

struct aml_area_system_binder aml_area_system_binder_mempolicy_generic = {
	aml_area_system_binder_mempolicy_pre_bind,
	aml_area_system_binder_mempolicy_post_bind,
	{0},
	0,
};

/*******************************************************************************
 * "System"-based areas:
 * Handle memory bindings and arenas using regular system interfaces, in a
 * pluggable way.
 ******************************************************************************/

struct aml_area_system_extra {
	struct aml_area_system_arena_mapper *mapper;
	struct aml_area_system_binder *binder;
};	

int aml_area_system_init(struct aml_area *area,
			 struct aml_area_system_mapper *mapper,
			 struct aml_area_system_binder *binder)
{
	assert(area != NULL);
	assert(mapper != NULL);
	assert(binder != NULL);
	struct aml_area_system_extra *myextra = malloc(sizeof(myextra));
	assert(myextra != NULL);
	myextra->mapper = mapper;
	myextra->binder = binder;
	area->extra = myextra;
	return 0;
}

int aml_area_system_destroy(struct aml_area *area)
{
	assert(area != NULL);
	assert(area->extra != NULL);
	free(area->extra);
	return 0;

}

struct aml_arena * aml_area_system_get_arena(struct aml_area *area)
{
	assert(area != NULL);
	struct aml_area_system_extra *extra = area->extra;
	return extra->mapper->get_arena(extra->mapper);
}

void *aml_area_system_mmap(struct aml_area *area, void *ptr, size_t sz)
{
	return mmap(ptr, sz, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS,
		    -1, 0);
}

int aml_area_system_pre_bind(struct aml_area *area, void *ptr, size_t sz)
{
	assert(area != NULL);
	struct aml_area_system_extra *extra = area->extra;
	return extra->binder->pre_bind(extra->binder);
}

int aml_area_system_post_bind(struct aml_area *area, void *ptr, size_t sz)
{
	assert(area != NULL);
	struct aml_area_system_extra *extra = area->extra;
	return extra->binder->post_bind(extra->binder, ptr, sz);
}

int aml_area_system_available(struct aml_area *area)
{
	return 1;
}

/*******************************************************************************
 * Area Templates
 * Area templates for typical types of areas.
 ******************************************************************************/

struct aml_area aml_area_system_default;
