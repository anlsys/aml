#include <aml.h>
#include <assert.h>
#include <sys/mman.h>

/*******************************************************************************
 * Regular Area
 * Handle memory allocation to DDR types of memory, no bindings whatsoever.
 ******************************************************************************/

int aml_area_regular_init(struct aml_area *area)
{
	assert(area != NULL);
	struct aml_arena *myarena = malloc(sizeof(struct aml_arena *));
	assert(myarena != NULL);
	area->extra = myarena;
	return aml_arena_init(myarena, &aml_arena_jemalloc, area);
}

int aml_area_regular_destroy(struct aml_area *area)
{
	assert(area != NULL);
	assert(area->extra != NULL);
	return aml_arena_destroy(area->extra);
}

struct aml_arena * aml_area_regular_get_arena(struct aml_area *area)
{
	return area->extra;
}

void *aml_area_regular_mmap(struct aml_area *area, void *ptr, size_t sz)
{
	return mmap(ptr, sz, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS,
		    -1, 0);
}

int aml_area_regular_mbind(struct aml_area *area, void *ptr, size_t sz)
{
	return 0;
}

int aml_area_regular_available(struct aml_area *area)
{
	return 1;
}

/*******************************************************************************
 * Area Templates
 * Area templates for typical types of areas.
 ******************************************************************************/

struct aml_area aml_area_hbm;
struct aml_area aml_area_regular = { aml_area_regular_init,
				     aml_area_regular_destroy,
				     aml_area_regular_get_arena,
				     aml_area_regular_mmap,
				     aml_area_regular_mbind,
				     aml_area_regular_available,
				     NULL};

/*******************************************************************************
 * Area Generic functions
 * Most of the stuff is dispatched to an arena, retrieved by type-specific
 * functions.
 ******************************************************************************/

int aml_area_init(struct aml_area *area, struct aml_area *template)
{
	assert(area != NULL);
	assert(template != NULL);
	/* copy template ops to area, then initialize it. */
	memcpy(area, template, sizeof(*area));
	return template->init(area);
}

int aml_area_destroy(struct aml_area *area)
{
	assert(area != NULL);
	return area->destroy(area);
}

void *aml_area_malloc(struct aml_area *area, size_t size)
{
	assert(area != NULL);
	struct aml_arena *arena = area->get_arena(area);
	assert(arena != NULL);
	return arena->malloc(arena, size);
}

void aml_area_free(struct aml_area *area, void *ptr)
{
	assert(area != NULL);
	struct aml_arena *arena = area->get_arena(area);
	assert(arena != NULL);
	arena->free(arena, ptr);
}

void *aml_area_calloc(struct aml_area *area, size_t num, size_t size)
{
	assert(area != NULL);
	struct aml_arena *arena = area->get_arena(area);
	assert(arena != NULL);
	return arena->calloc(arena, num, size);
}

void *aml_area_realloc(struct aml_area *area, void *ptr, size_t size)
{
	assert(area != NULL);
	struct aml_arena *arena = area->get_arena(area);
	assert(arena != NULL);
	return arena->realloc(arena, ptr, size);
}

void *aml_area_acquire(struct aml_area *area, size_t size)
{
	assert(area != NULL);
	struct aml_arena *arena = area->get_arena(area);
	assert(arena != NULL);
	return arena->acquire(arena,size);
}

void aml_area_release(struct aml_area *area, void *ptr)
{
	assert(area != NULL);
	struct aml_arena *arena = area->get_arena(area);
	assert(arena != NULL);
	arena->release(arena, ptr);
}

