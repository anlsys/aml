#include <aml.h>
#include <assert.h>

int main(int argc, char *argv[])
{
	struct aml_vector v;

	/* no need for library initialization */
	;

	/* struct to test the offsets */
	struct test {
		unsigned long unused;
		int key;
	};
	assert(!aml_vector_init(&v, 1, sizeof(struct test),
				offsetof(struct test, key), -1));

	/* assert the size */
	assert(aml_vector_size(&v) == 1);

	/* add an element and look for some */
	struct test *e = aml_vector_get(&v, 0);
	assert(e != NULL);
	e->unused = 42;
	e->key = 24;
	assert(aml_vector_find(&v, 24) == 0);
	assert(aml_vector_find(&v, 42) == -1);

	/* add a second element, trigger a resize, and check it */
	struct test *f = aml_vector_add(&v);
	assert(f != NULL && f->key == -1);
	assert(aml_vector_find(&v, 42) == -1);
	assert(aml_vector_find(&v, -1) == 1);
	assert(aml_vector_size(&v) == 2);

	aml_vector_destroy(&v);
	return 0;
}
