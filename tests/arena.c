#include <aml.h>
#include <assert.h>

int main()
{
	struct aml_arena arena;
	assert(!aml_arena_init(&arena, &aml_arena_jemalloc, &aml_area_regular));
	assert(!aml_arena_destroy(&arena));
	return 0;
}
