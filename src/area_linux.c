#include <aml.h>
#include <assert.h>
#include <sys/mman.h>

/*******************************************************************************
 * Linux backed area:
 * The area itself is organized into various two sets of data:
 * - parameters that defines what policies are applied
 * - methods that implement those policies
 * This is designed similarly to component-entity-systems.
 ******************************************************************************/

/* two functions used by arenas to handle real allocations */

void *aml_area_linux_mmap(struct aml_area_data *a, void *ptr, size_t sz)
{
	assert(a != NULL);
	void *ret = NULL;
	struct aml_area_linux *area = (struct aml_area_linux *)a;
	area->ops.mbind.pre_bind(&area->data.mbind);
	ret = area->ops.mmap.mmap(&area->data.mmap, ptr, sz);
	area->ops.mbind.post_bind(&area->data.mbind, ret, sz);
	return ret;
}

int aml_area_linux_available(struct aml_area_data *a)
{
	return 1;
}

/*******************************************************************************
 * Public API:
 * The actual functions that will be called on the area from users
 * Respect the POSIX spec for those functions.
 ******************************************************************************/

void *aml_area_linux_malloc(struct aml_area_data *a, size_t size)
{
	assert(a != NULL);
	struct aml_area_linux *area = (struct aml_area_linux *)a;
	struct aml_arena *arena = (struct aml_arena *)
		area->ops.manager.get_arena(&area->data.manager);
	assert(arena != NULL);
	if(size == 0)
		return NULL;
	return aml_arena_mallocx(arena, size, AML_ARENA_FLAG_ZERO);
}

void aml_area_linux_free(struct aml_area_data *a, void *ptr)
{
	assert(a != NULL);
	struct aml_area_linux *area = (struct aml_area_linux *)a;
	struct aml_arena *arena = (struct aml_arena *)
		area->ops.manager.get_arena(&area->data.manager);
	assert(arena != NULL);
	if(ptr == NULL)
		return;
	aml_arena_dallocx(arena, ptr, 0);
}

void *aml_area_linux_calloc(struct aml_area_data *a, size_t num, size_t size)
{
	assert(a != NULL);
	return aml_area_linux_malloc(a, num*size);
}

void *aml_area_linux_realloc(struct aml_area_data *a, void *ptr, size_t size)
{
	assert(a != NULL);
	struct aml_area_linux *area = (struct aml_area_linux *)a;
	if(ptr == NULL)
		return aml_area_linux_malloc(a, size);
	struct aml_arena *arena = (struct aml_arena *)
		area->ops.manager.get_arena(&area->data.manager);
	assert(arena != NULL);
	if(size == 0)
	{
		aml_arena_dallocx(arena, ptr, 0);
		return NULL;
	}
	return aml_arena_reallocx(arena, ptr, size, 0);
}

void *aml_area_linux_acquire(struct aml_area_data *a, size_t size)
{
	assert(a != NULL);
	struct aml_area_linux *area = (struct aml_area_linux *)a;
	struct aml_arena *arena = (struct aml_arena *)
		area->ops.manager.get_arena(&area->data.manager);
	assert(arena != NULL);
	return aml_arena_mallocx(arena, size, 0);
}

void aml_area_linux_release(struct aml_area_data *a, void *ptr)
{
	assert(a != NULL);
	struct aml_area_linux *area = (struct aml_area_linux *)a;
	struct aml_arena *arena = (struct aml_arena *)
		area->ops.manager.get_arena(&area->data.manager);
	assert(arena != NULL);
	aml_arena_dallocx(arena, ptr, 0);
}

struct aml_area_ops aml_area_linux_ops = {
	aml_area_linux_malloc,
	aml_area_linux_free,
	aml_area_linux_calloc,
	aml_area_linux_realloc,
	aml_area_linux_acquire,
	aml_area_linux_release,
	aml_area_linux_mmap,
	aml_area_linux_available,
};

/*******************************************************************************
 * Initialization/Destroy function:
 * Collections of init/destroy functions for popular types of linux-based areas
 ******************************************************************************/

int aml_area_linux_init(struct aml_area_linux *area)
{
	assert(area != NULL);
	return 0;
}

int aml_area_linux_destroy(struct aml_area_linux *area)
{
	assert(area != NULL);
	return 0;
}
