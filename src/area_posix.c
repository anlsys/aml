#include <aml.h>
#include <assert.h>
#include <sys/mman.h>

/*******************************************************************************
 * POSIX backed area:
 * This area just calls regular posix functions. No magic.
 ******************************************************************************/

/* two functions used by arenas to handle real allocations. Note that unless
 * arenas are created explicitly with area_posix as a backend, these functions
 * will never be used, as we don't use arenas in this area.*/

void *aml_area_posix_mmap(struct aml_area_data *data, void *ptr, size_t sz)
{
	return mmap(ptr, sz, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS,
		    -1, 0);
}

int aml_area_posix_available(struct aml_area_data *data)
{
	assert(data != NULL);
	return 1;
}

/*******************************************************************************
 * Public API:
 * The actual functions that will be called on the area from users
 * Respect the POSIX spec for those functions.
 ******************************************************************************/

void *aml_area_posix_malloc(struct aml_area_data *data, size_t size)
{
	assert(data != NULL);
	return malloc(size);
}

void aml_area_posix_free(struct aml_area_data *data, void *ptr)
{
	assert(data != NULL);
	free(ptr);
}

void *aml_area_posix_calloc(struct aml_area_data *data, size_t nm, size_t size)
{
	assert(data != NULL);
	return calloc(nm, size);
}

void *aml_area_posix_realloc(struct aml_area_data *data, void *ptr, size_t size)
{
	assert(data != NULL);
	return realloc(ptr, size);
}

void *aml_area_posix_acquire(struct aml_area_data *data, size_t size)
{
	assert(data != NULL);
	return malloc(size);
}

void aml_area_posix_release(struct aml_area_data *data, void *ptr)
{
	assert(data != NULL);
	free(ptr);
}

struct aml_area_ops aml_area_posix_ops = {
	aml_area_posix_malloc,
	aml_area_posix_free,
	aml_area_posix_calloc,
	aml_area_posix_realloc,
	aml_area_posix_acquire,
	aml_area_posix_release,
	aml_area_posix_mmap,
	aml_area_posix_available,
};

/*******************************************************************************
 * Initialization/Destroy function:
 ******************************************************************************/

int aml_area_posix_create(struct aml_area **a)
{
	struct aml_area *ret = NULL;
	intptr_t baseptr, dataptr;

	/* alloc */
	baseptr = (intptr_t) calloc(1, AML_AREA_POSIX_ALLOCSIZE);
	dataptr = baseptr + sizeof(struct aml_area);

	ret = (struct aml_area *)baseptr;
	ret->data = (struct aml_area_data *)dataptr;

	aml_area_posix_vinit(ret);

	*a = ret;
	return 0;
}

int aml_area_posix_vinit(struct aml_area *data)
{
	assert(data != NULL);
	return 0;
}

int aml_area_posix_init(struct aml_area *data)
{
	assert(data != NULL);
	return 0;
}

int aml_area_posix_destroy(struct aml_area *data)
{
	assert(data != NULL);
	return 0;
}
