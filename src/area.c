#include <aml.h>
#include <assert.h>

/*******************************************************************************
 * Area Generic functions
 * Most of the stuff is dispatched to a different layer, using type-specific
 * functions.
 ******************************************************************************/

void *aml_area_malloc(struct aml_area *area, size_t size)
{
	assert(area != NULL);
	return area->ops->malloc(area->data, size);
}

void aml_area_free(struct aml_area *area, void *ptr)
{
	assert(area != NULL);
	area->ops->free(area->data, ptr);
}

void *aml_area_calloc(struct aml_area *area, size_t num, size_t size)
{
	assert(area != NULL);
	return area->ops->calloc(area->data, num, size);
}

void *aml_area_realloc(struct aml_area *area, void *ptr, size_t size)
{
	assert(area != NULL);
	return area->ops->realloc(area->data, ptr, size);
}

void *aml_area_acquire(struct aml_area *area, size_t size)
{
	assert(area != NULL);
	return area->ops->acquire(area->data, size);
}

void aml_area_release(struct aml_area *area, void *ptr)
{
	assert(area != NULL);
	area->ops->release(area->data, ptr);
}

void *aml_area_mmap(struct aml_area *area, void *ptr, size_t size)
{
	assert(area != NULL);
	return area->ops->mmap(area->data, ptr, size);
}

int aml_area_available(const struct aml_area *area)
{
	assert(area != NULL);
	return area->ops->available(area->data);
}

int aml_area_binding(const struct aml_area *area, struct aml_binding **binding)
{
	assert(area != NULL);
	return area->ops->binding(area->data, binding);
}
