#include <aml.h>
#include <assert.h>
#include <inttypes.h>
#include <numa.h>
#include <numaif.h>
#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>

#define PAGE_SIZE 4096
#define ALLOC_SIZE (1<<20)
#define ALLOC_PAGES (ALLOC_SIZE/PAGE_SIZE)
#define ARRAY_SIZE(x) (sizeof(x)/sizeof(x[0]))

/* should check that all mmap calls match the expectations of arenas:
 * - respect POSIX as much as possible
 * - valid pointers returned for arenas
 */
void doit(struct aml_area_linux_mmap_data *config)
{
	int err;
	void *ptr;

	ptr = aml_area_linux_mmap_generic(config, NULL, 0);
	assert(ptr == MAP_FAILED);

	ptr = aml_area_linux_mmap_generic(config, NULL, ALLOC_SIZE);
	assert(ptr != MAP_FAILED);

	/* should crash if the alloc was bad */
	memset(ptr, 42, ALLOC_SIZE);

	/* alignments */
	assert((uintptr_t)ptr % PAGE_SIZE == 0);

	munmap(ptr, ALLOC_SIZE);
}

int main(int argc, char *argv[])
{
	struct aml_area_linux_mmap_data config[3];
	char tmpname[] = "aml.mmap.test.XXXXXX";
	char tmpname2[] = "aml.mmap.test.XXXXXX";
	int tmpfile;
	aml_init(&argc, &argv);

	aml_area_linux_mmap_anonymous_init(&config[0]);

	tmpfile = mkstemp(tmpname);
	assert(tmpfile != -1);
	/* use ftruncate to make sure we have enough space in the file. */
	ftruncate(tmpfile, ALLOC_SIZE);
	aml_area_linux_mmap_fd_init(&config[1], tmpfile, ALLOC_SIZE);

	aml_area_linux_mmap_tmpfile_init(&config[2], tmpname2, ALLOC_SIZE);

	for(int i = 0; i < ARRAY_SIZE(config); i++)
		doit(&config[i]);

	aml_area_linux_mmap_tmpfile_destroy(&config[2]);
	aml_area_linux_mmap_fd_destroy(&config[1]);
	assert(!close(tmpfile));
	assert(!unlink(tmpname));
	aml_area_linux_mmap_anonymous_destroy(&config[0]);
	assert(!unlink(tmpname2));

	aml_finalize();
	return 0;
}
