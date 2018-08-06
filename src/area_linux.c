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
	struct aml_area_linux *area = (struct aml_area_linux *)a;
	return area->ops.mmap.mmap(&area->data.mmap, ptr, sz);
}

int aml_area_linux_available(const struct aml_area_data *a)
{
	return 1;
}

int aml_area_linux_binding(const struct aml_area_data *a, struct aml_binding **b)
{
	assert(a != NULL);
	const struct aml_area_linux *area = (const struct aml_area_linux *)a;
	return area->ops.mbind.binding(&area->data.mbind, b);
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
	void *ret;
	assert(arena != NULL);
	if(size == 0)
		return NULL;

	area->ops.mbind.pre_bind(&area->data.mbind);
	ret = aml_arena_mallocx(arena, size, AML_ARENA_FLAG_ZERO);
	area->ops.mbind.post_bind(&area->data.mbind, ret, size);
	return ret;
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

	area->ops.mbind.pre_bind(&area->data.mbind);
	ptr = aml_arena_reallocx(arena, ptr, size, 0);
	area->ops.mbind.post_bind(&area->data.mbind, ptr, size);
	return ptr;
}

void *aml_area_linux_acquire(struct aml_area_data *a, size_t size)
{
	assert(a != NULL);
	struct aml_area_linux *area = (struct aml_area_linux *)a;
	struct aml_arena *arena = (struct aml_arena *)
		area->ops.manager.get_arena(&area->data.manager);
	void *ret;
	assert(arena != NULL);
	area->ops.mbind.pre_bind(&area->data.mbind);
	ret = aml_arena_mallocx(arena, size, 0);
	area->ops.mbind.post_bind(&area->data.mbind, ret, size);
	return ret;
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
	aml_area_linux_binding,
};

/*******************************************************************************
 * Initialization/Destroy function:
 * Collections of init/destroy functions for popular types of linux-based areas
 ******************************************************************************/

int aml_area_linux_create(struct aml_area **a, int manager_type,
			  int mbind_type, int mmap_type, ...)
{
	va_list ap;
	struct aml_area *ret = NULL;
	intptr_t baseptr, dataptr;
	va_start(ap, mmap_type);

	/* alloc */
	baseptr = (intptr_t) calloc(1, AML_AREA_LINUX_ALLOCSIZE);
	dataptr = baseptr + sizeof(struct aml_area);

	ret = (struct aml_area *)baseptr;
	ret->data = (struct aml_area_data *)dataptr;

	aml_area_linux_vinit(ret, manager_type, mbind_type, mmap_type, ap);

	va_end(ap);
	*a = ret;
	return 0;

}

int aml_area_linux_vinit(struct aml_area *a, int manager_type,
			 int mbind_type, int mmap_type, va_list ap)
{
	a->ops = &aml_area_linux_ops;
	struct aml_area_linux *area = (struct aml_area_linux *)a->data;

	/* manager init */
	assert(manager_type == AML_AREA_LINUX_MANAGER_TYPE_SINGLE);
	struct aml_arena *arena = va_arg(ap, struct aml_arena *);
	aml_area_linux_manager_single_init(&area->data.manager, arena);
	area->ops.manager = aml_area_linux_manager_single_ops;

	/* mbind init */
	int policy = va_arg(ap, int);
	unsigned long *nodemask = va_arg(ap, unsigned long *);
	aml_area_linux_mbind_init(&area->data.mbind, policy, nodemask);
	if(mbind_type == AML_AREA_LINUX_MBIND_TYPE_REGULAR)
		area->ops.mbind = aml_area_linux_mbind_regular_ops;
	else if(mbind_type == AML_AREA_LINUX_MBIND_TYPE_MEMPOLICY)
		area->ops.mbind = aml_area_linux_mbind_mempolicy_ops;

	/* mmap init */
	area->ops.mmap = aml_area_linux_mmap_generic_ops;
	if(mmap_type == AML_AREA_LINUX_MMAP_TYPE_ANONYMOUS)
		aml_area_linux_mmap_anonymous_init(&area->data.mmap);
	else if(mmap_type == AML_AREA_LINUX_MMAP_TYPE_FD)
	{
		int fd = va_arg(ap, int);
		off_t offset = va_arg(ap, off_t);
		aml_area_linux_mmap_fd_init(&area->data.mmap, fd, offset);
	}
	else if(mmap_type == AML_AREA_LINUX_MMAP_TYPE_TMPFILE)
	{
		char *template = va_arg(ap, char*);
		size_t max = va_arg(ap, size_t);
		aml_area_linux_mmap_tmpfile_init(&area->data.mmap, template,
						 max);
	}
	aml_arena_register(arena, a);
	return 0;
}

int aml_area_linux_init(struct aml_area *a, int manager_type,
			int mbind_type, int mmap_type, ...)
{
	int err;
	va_list ap;
	va_start(ap, mmap_type);
	err = aml_area_linux_vinit(a, manager_type, mbind_type, mmap_type, ap);
	va_end(ap);
	return err;

}

int aml_area_linux_destroy(struct aml_area *a)
{
	struct aml_area_linux *area = (struct aml_area_linux *)a->data;
	struct aml_arena *arena = area->data.manager.pool;
	aml_arena_deregister(arena);
	aml_area_linux_mmap_anonymous_destroy(&area->data.mmap);
	aml_area_linux_mbind_destroy(&area->data.mbind);
	aml_area_linux_manager_single_destroy(&area->data.manager);
	aml_arena_jemalloc_destroy(arena);
	return 0;
}
