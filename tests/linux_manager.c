#include <aml.h>
#include <assert.h>

void doit(struct aml_area_linux_manager_data *data,
	  struct aml_area_linux_manager_ops *ops)
{
	struct aml_arena *arena;
	void *ptr;
	unsigned long *a;

	arena = ops->get_arena(data);
	assert(arena != NULL);

	/* try to allocate something */
	ptr = aml_arena_mallocx(arena, sizeof(unsigned long) *10, 0);
	assert(ptr != NULL);
	a = (unsigned long *)ptr;
	assert(a[0] == 0);
	assert(a[0] == a[9]);
	aml_arena_dallocx(arena, ptr, 0);
}

#define ARRAY_SIZE(x) (sizeof(x)/sizeof(x[0]))

int main(int argc, char *argv[])
{
	struct aml_area area;
	struct aml_area_posix_data area_data;
	AML_ARENA_JEMALLOC_DECL(arena);
	struct aml_area_linux_manager_data config[1];

	aml_init(&argc, &argv);

	/* init all the necessary objects:
	 * we use a posix area to provide mmap, and jemalloc arena as the
	 * managed arena.*/
	aml_area_posix_init(&area_data);
	area.ops = &aml_area_posix_ops;
	area.data = (struct aml_area_data *)&area_data;
	assert(!aml_arena_jemalloc_init(&arena, AML_ARENA_JEMALLOC_TYPE_REGULAR));
	assert(!aml_arena_register(&arena, &area));

	aml_area_linux_manager_single_init(&config[0], &arena);

	for(int i = 0; i < ARRAY_SIZE(config); i++)
		doit(&config[i], &aml_area_linux_manager_single_ops);

	aml_area_linux_manager_single_destroy(&config[0]);
	assert(!aml_arena_deregister(&arena));
	aml_arena_jemalloc_destroy(&arena);
	aml_area_posix_init(&area_data);
	aml_finalize();
	return 0;
}
