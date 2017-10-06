#include <aml.h>
#include <assert.h>

int main(int argc, char *argv[])
{
	struct aml_area area;
	void *ptr;
	unsigned long *a, *b, *c;
	unsigned long i;

	/* library initialization */
	aml_init(&argc, &argv);

	/* initialize the area itself */
	aml_area_init(&area, &aml_area_regular);

	/* try to allocate something */
	ptr = aml_area_malloc(&area, sizeof(unsigned long) * 10);
	assert(ptr != NULL);
	a = (unsigned long *)ptr;
	assert(a[0] == 0);
	assert(a[0] == a[9]);
	aml_area_free(&area, ptr);

	/* libc API compatibility: malloc(0) */
	ptr = aml_area_malloc(&area, 0);
	assert(ptr == NULL);
	aml_area_free(&area, ptr);

	/* calloc */
	ptr = aml_area_calloc(&area, 10, sizeof(unsigned long));
	assert(ptr != NULL);
	a = (unsigned long *)ptr;
	assert(a[0] == 0);
	assert(a[0] == a[9]);
	aml_area_free(&area, ptr);

	/* libc API compatibility: calloc(0) */
	ptr = aml_area_calloc(&area, 0, sizeof(unsigned long));
	assert(ptr == NULL);
	ptr = aml_area_calloc(&area, 10, 0);
	assert(ptr == NULL);

	/* realloc */
	ptr = aml_area_realloc(&area, NULL, sizeof(unsigned long) * 10);
	assert(ptr != NULL);
	ptr = aml_area_realloc(&area, ptr, sizeof(unsigned long) * 2);
	assert(ptr != NULL);
	ptr = aml_area_realloc(&area, ptr, sizeof(unsigned long) * 20);
	assert(ptr != NULL);
	ptr = aml_area_realloc(&area, ptr, 0);

	aml_area_destroy(&area);
	aml_finalize();
	return 0;
}