#include <aml.h>
#include <assert.h>
#include <numa.h>
#include <numaif.h>

void doit(struct aml_area *area)
{
	void *ptr;
	unsigned long *a, *b, *c;

	/* try to allocate something */
	ptr = aml_area_malloc(area, sizeof(unsigned long) * 10);
	assert(ptr != NULL);
	a = (unsigned long *)ptr;
	memset(a, 0, sizeof(unsigned long)*10);
	assert(a[0] == 0);
	assert(a[0] == a[9]);
	aml_area_free(area, ptr);

	/* libc API compatibility: malloc(0):
	 * returns either null or unique valid for free. */
	ptr = aml_area_malloc(area, 0);
	aml_area_free(area, ptr);

	/* calloc */
	ptr = aml_area_calloc(area, 10, sizeof(unsigned long));
	assert(ptr != NULL);
	a = (unsigned long *)ptr;
	assert(a[0] == 0);
	assert(a[0] == a[9]);
	aml_area_free(area, ptr);

	/* libc API compatibility: calloc(0): same as malloc(0) */
	ptr = aml_area_calloc(area, 0, sizeof(unsigned long));
	aml_area_free(area, ptr);
	ptr = aml_area_calloc(area, 10, 0);
	aml_area_free(area, ptr);

	/* realloc */
	ptr = aml_area_realloc(area, NULL, sizeof(unsigned long) * 10);
	assert(ptr != NULL);
	ptr = aml_area_realloc(area, ptr, sizeof(unsigned long) * 2);
	assert(ptr != NULL);
	ptr = aml_area_realloc(area, ptr, sizeof(unsigned long) * 20);
	assert(ptr != NULL);
	ptr = aml_area_realloc(area, ptr, 0);
}

#define ARRAY_SIZE(x) (sizeof(x)/sizeof(x[0]))

int main(int argc, char *argv[])
{
	struct aml_area area;
	struct aml_arena_jemalloc_data arena_data;
	struct aml_arena arena = {&aml_arena_jemalloc_ops,
		(struct aml_arena_data *)&arena_data};
	struct aml_area_linux area_data;
	unsigned long nodemask[AML_NODEMASK_SZ];
	struct bitmask *allowed;

	/* library initialization */
	aml_init(&argc, &argv);

	/* initialize the area itself */
	assert(!aml_area_linux_init(&area_data));
	area.ops = &aml_area_linux_ops;
	area.data = (struct aml_area_data*)&area_data;

	/* ops init */
	area_data.ops.manager = aml_area_linux_manager_single_ops;
	area_data.ops.mbind = aml_area_linux_mbind_regular_ops;
	area_data.ops.mmap = aml_area_linux_mmap_generic_ops;

	/* init all the inner objects:
	 * WARNING: there an order to this madness. */
	assert(!aml_arena_jemalloc_regular_init(&arena_data));
	assert(!aml_area_linux_manager_single_init(&area_data.data.manager,
						   &arena));
	allowed = numa_get_mems_allowed();
	memcpy(nodemask, allowed->maskp, AML_NODEMASK_BYTES);
	assert(!aml_area_linux_mbind_init(&area_data.data.mbind, MPOL_BIND,
					  nodemask));
	assert(!aml_area_linux_mmap_anonymous_init(&area_data.data.mmap));
	assert(!aml_arena_register(&arena, &area));

	doit(&area);

	/* same here, order matters. */
	assert(!aml_arena_deregister(&arena));
	assert(!aml_area_linux_mmap_anonymous_destroy(&area_data.data.mmap));
	assert(!aml_area_linux_mbind_destroy(&area_data.data.mbind));
	assert(!aml_area_linux_manager_single_destroy(&area_data.data.manager));
	assert(!aml_arena_jemalloc_regular_destroy(&arena_data));
	assert(!aml_area_linux_destroy(&area_data));

	aml_finalize();
	return 0;
}
