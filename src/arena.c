#include <aml.h>
#include <assert.h>

/*******************************************************************************
 * Arena Generic functions
 * Most of the stuff is dispatched to a different layer, using type-specific
 * functions.
 ******************************************************************************/

int aml_arena_register(struct aml_arena *arena, struct aml_area *area)
{
	assert(arena != NULL);
	assert(area != NULL);
	return arena->ops->register_arena(arena->data, area);
}

int aml_arena_deregister(struct aml_arena *arena)
{
	assert(arena != NULL);
	return arena->ops->deregister_arena(arena->data);
}

void *aml_arena_mallocx(struct aml_arena *arena, size_t size, int flags)
{
	assert(arena != NULL);
	if(size == 0)
		return NULL;
	return arena->ops->mallocx(arena->data, size, flags);
}

void aml_arena_dallocx(struct aml_arena *arena, void *ptr, int flags)
{
	assert(arena != NULL);
	if(ptr == NULL)
		return;
	arena->ops->dallocx(arena->data, ptr, flags);
}

void *aml_arena_reallocx(struct aml_arena *arena, void *ptr, size_t size,
			 int flags)
{
	assert(arena != NULL);
	if(size == 0)
	{
		if(ptr != NULL)
			arena->ops->dallocx(arena->data, ptr, flags);
		return NULL;
	}
	if(ptr == NULL)
		return arena->ops->mallocx(arena->data, size, flags);
	return arena->ops->reallocx(arena->data, ptr, size, flags);
}
