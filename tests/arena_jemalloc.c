#include <aml.h>
#include <assert.h>

#define ARRAY_SIZE(x) (sizeof(x)/sizeof(x[0]))

/* posix area used as a backend */
AML_AREA_POSIX_DECL(area);

void doit(struct aml_arena *arena)
{
	int err;
	void *ptr;
	unsigned long *a;

	/* create the arena */
	err = aml_arena_register(arena, &area);
	assert(err == 0);

	/* try to allocate something */
	ptr = aml_arena_mallocx(arena, sizeof(unsigned long) * 10,
				AML_ARENA_FLAG_ZERO);
	assert(ptr != NULL);
	a = (unsigned long *)ptr;
	assert(a[0] == 0);
	assert(a[0] == a[9]);
	aml_arena_dallocx(arena, ptr, 0);

	/* libc API compatibility: malloc(0):
	 * returns either null or unique valid for free. */
	ptr = aml_arena_mallocx(arena, 0, 0);
	aml_arena_dallocx(arena, ptr, 0);

	/* realloc */
	ptr = aml_arena_reallocx(arena, NULL, sizeof(unsigned long) * 10, 0);
	assert(ptr != NULL);
	ptr = aml_arena_reallocx(arena, ptr, sizeof(unsigned long) * 2, 0);
	assert(ptr != NULL);
	ptr = aml_arena_reallocx(arena, ptr, sizeof(unsigned long) * 20, 0);
	assert(ptr != NULL);
	ptr = aml_arena_reallocx(arena, ptr, 0, 0);

	err = aml_arena_deregister(arena);
	assert(err == 0);
}

int main(int argc, char *argv[])
{
	struct aml_arena *arenas[4];
	/* library init */
	aml_init(&argc, &argv);

	/* area init */
	aml_area_posix_init(&area);

	/* build up the data variants */
	assert(!aml_arena_jemalloc_create(&arenas[0],
					  AML_ARENA_JEMALLOC_TYPE_REGULAR));
	assert(!aml_arena_jemalloc_create(&arenas[1],
					  AML_ARENA_JEMALLOC_TYPE_ALIGNED,
					  (size_t)42));
	assert(!aml_arena_jemalloc_create(&arenas[2],
					  AML_ARENA_JEMALLOC_TYPE_GENERIC,
					  arenas[0]->data));
	/* alignment bigger than PAGE_SIZE */
	assert(!aml_arena_jemalloc_create(&arenas[3],
					  AML_ARENA_JEMALLOC_TYPE_ALIGNED, 4100));

	for(int i = 0; i < ARRAY_SIZE(arenas); i++)
		doit(arenas[i]);

	for(int i = 0; i < ARRAY_SIZE(arenas); i++)
	{
		assert(!aml_arena_jemalloc_destroy(arenas[i]));
		free(arenas[i]);
	}
	aml_area_posix_destroy(&area);
	return 0;
}
