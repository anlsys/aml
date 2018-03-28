#include <aml.h>
#include <assert.h>

#define TILESIZE (2)
#define NBTILES (4)

int main(int argc, char *argv[])
{
	AML_BINDING_SINGLE_DECL(binding);
	AML_TILING_1D_DECL(tiling);
	AML_ARENA_JEMALLOC_DECL(arena);
	AML_AREA_LINUX_DECL(area);
	AML_DMA_LINUX_PAR_DECL(dma);
	unsigned long nodemask[AML_NODEMASK_SZ];
	void *dst, *src;

	/* library initialization */
	aml_init(&argc, &argv);

	/* initialize all the supporting struct */
	assert(!aml_binding_init(&binding, AML_BINDING_TYPE_SINGLE, 0));
	assert(!aml_tiling_init(&tiling, AML_TILING_TYPE_1D, TILESIZE*PAGE_SIZE,
				TILESIZE*PAGE_SIZE*NBTILES));
	AML_NODEMASK_ZERO(nodemask);
	AML_NODEMASK_SET(nodemask, 0);
	assert(!aml_arena_jemalloc_init(&arena, AML_ARENA_JEMALLOC_TYPE_REGULAR));

	assert(!aml_area_linux_init(&area,
				    AML_AREA_LINUX_MANAGER_TYPE_SINGLE,
				    AML_AREA_LINUX_MBIND_TYPE_REGULAR,
				    AML_AREA_LINUX_MMAP_TYPE_ANONYMOUS,
				    &arena, MPOL_BIND, nodemask));

	size_t maxrequests = NBTILES;
	size_t maxthreads = 4;
	assert(!aml_dma_linux_par_init(&dma, maxrequests, maxthreads));

	/* allocate some memory */
	src = aml_area_malloc(&area, TILESIZE*PAGE_SIZE*NBTILES);
	assert(src != NULL);
	dst = aml_area_malloc(&area, TILESIZE*PAGE_SIZE*NBTILES);
	assert(dst != NULL);

	memset(src, 42, TILESIZE*PAGE_SIZE*NBTILES);
	memset(dst, 24, TILESIZE*PAGE_SIZE*NBTILES);

	/* move some stuff by copy */
	for(int i = 0; i < NBTILES; i++)
		aml_dma_copy(&dma, &tiling, dst, i, &tiling, src, i);

	assert(!memcmp(src, dst, TILESIZE*PAGE_SIZE*NBTILES));

	/* now move it by pages */
	for(int i = 0; i < NBTILES; i++)
		aml_dma_move(&dma, &area, &tiling, src, i);

	/* delete everything */
	aml_dma_linux_par_destroy(&dma);
	aml_area_free(&area, dst);
	aml_area_free(&area, src);
	aml_area_linux_destroy(&area);
	aml_tiling_destroy(&tiling, AML_TILING_TYPE_1D);
	aml_binding_destroy(&binding, AML_BINDING_TYPE_SINGLE);

	aml_finalize();
	return 0;
}