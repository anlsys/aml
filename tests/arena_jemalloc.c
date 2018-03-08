#include <aml.h>
#include <assert.h>

#define ARRAY_SIZE(x) (sizeof(x)/sizeof(x[0]))

/* posix area used as a backend */
struct aml_area area;
struct aml_area_posix_data posix_data;


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
	struct aml_arena_jemalloc_data data[4];
	struct aml_arena arenas[] = {
		{&aml_arena_jemalloc_ops, (struct aml_arena_data *)&data[0]},
		{&aml_arena_jemalloc_ops, (struct aml_arena_data *)&data[1]},
		{&aml_arena_jemalloc_ops, (struct aml_arena_data *)&data[2]},
		{&aml_arena_jemalloc_ops, (struct aml_arena_data *)&data[3]},
	};
	/* library init */
	aml_init(&argc, &argv);

	/* area init */
	aml_area_posix_init(&posix_data);
	area.ops = &aml_area_posix_ops;
	area.data = (struct aml_area_data *)&posix_data;

	/* build up the data variants */
	assert(!aml_arena_jemalloc_regular_init(&data[0]));
	assert(!aml_arena_jemalloc_aligned_init(&data[1], 42));
	assert(!aml_arena_jemalloc_generic_init(&data[2], &data[0]));
	/* alignment bigger than PAGE_SIZE */
	assert(!aml_arena_jemalloc_aligned_init(&data[3], 4100));

	for(int i = 0; i < ARRAY_SIZE(arenas); i++)
		doit(&arenas[i]);

	assert(!aml_arena_jemalloc_regular_destroy(&data[0]));
	assert(!aml_arena_jemalloc_align_destroy(&data[1]));
	assert(!aml_arena_jemalloc_generic_destroy(&data[2]));
	assert(!aml_arena_jemalloc_align_destroy(&data[3]));
	return 0;
}
