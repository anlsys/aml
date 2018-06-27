#include <aml.h>
#include <assert.h>
#include <errno.h>
#include <omp.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define ITER 10
#define MEMSIZE 131072//67108864//1024 entries by 1024 entries * sizeof(unsigned long)

size_t numthreads;
//size of 2D Tiles in A matrix
size_t tilesz, esz;
unsigned long CHUNKING;
unsigned long *a, *b, *c;
AML_TILING_2D_DECL(tiling);
AML_AREA_LINUX_DECL(slow);
AML_AREA_LINUX_DECL(fast);
AML_SCRATCH_PAR_DECL(sa);
AML_SCRATCH_PAR_DECL(sb);


//TODO
int kernel(unsigned long *a, unsigned long *b, unsigned long *c, size_t n)
{
	size_t i;
	//printf("%p = %p + %p [%zi]\n",c,a,b,n);
	for(i = 0; i < n; i++)
		c[0] += a[i] * b[i];
	return 0;
}

//TODO
void do_work(unsigned long tid)
{

	int offset, i, j, ai, bi, oldai, oldbi;
	unsigned long *ap, *bp, *cp;
	void *abaseptr, *bbaseptr;
	offset = tid*CHUNKING;
	ap = aml_tiling_tilestart(&tiling, a, offset);
	bp = aml_tiling_tilestart(&tiling, b, 0);
	cp = aml_tiling_tilestart(&tiling, c, offset);
	abaseptr = aml_scratch_baseptr(&sa);
	bbaseptr = aml_scratch_baseptr(&sb);
	ai = -1; bi = -1;

	//This double for loop will have each thread iterate for different rows of C for CHUNKING number of rows.
	//It starts by async pulling the next row in a while we work on the current row of A.
	//Then it begins an inner loop of pulling the next row of Transposed B while working on current row of B
	//Then it jumps into the kernel and dot products A row and B row and accumulates the result in the respective C location.
	//Then it returns and loops until an entire row for C is done.
	//End inner loop
	//The code then waits to begin the next given chunk (wait on &sa)
	//Resets the tile start positions to get next row of C and A and getting first column of B again.
	//Run one time more to do last rows
	for(i = 0; i < CHUNKING-1; i++) {
		struct aml_scratch_request *ar, *br;
		oldai = ai; 
		aml_scratch_async_pull(&sa, &ar, abaseptr, &ai, a, offset+i+1);
		for (j = 0; j < esz; j++)
		{
			oldbi = bi;
			aml_scratch_async_pull(&sb, &br, bbaseptr, &bi, b, offset+i+1);
			//This will have cp be a pointer to the exact spot in memory that the row by column multiplication will happen.
			if(tid == 0){
				//printf("Pointer is %p", cp + j);
			}
			kernel(ap, bp, cp + j, esz);
			aml_scratch_wait(&sb, br);
			bp = aml_tiling_tilestart(&tiling, bbaseptr, 0);
			aml_scratch_release(&sb, oldbi);
		}
		aml_scratch_wait(&sa, ar);
		ap = aml_tiling_tilestart(&tiling, abaseptr, ai);
		cp = aml_tiling_tilestart(&tiling, c, offset+i+1);
		aml_scratch_release(&sa, oldai);
	}
	//Third argument may be wrong
	for (j = 0; j < esz; j++){
		kernel(ap, bp, cp + j, esz);	
	}
	//kernel(ap, bp, cp + CHUNKING - 1, esz);


}


//The first approach is going to be a simple row x column approach. Then once an actual matrix multiplication program
//as been created, a tiling apporach can be made. I will use the 2D library initially, but it will be like a 1D tile
int main(int argc, char *argv[])
{
	AML_BINDING_SINGLE_DECL(binding);
	AML_ARENA_JEMALLOC_DECL(arena);
	AML_DMA_LINUX_SEQ_DECL(dma);
	unsigned long nodemask[AML_NODEMASK_SZ];
	aml_init(&argc, &argv);
	assert(argc == 1);

	omp_set_num_threads(64);
	/* use openmp env to figure out how many threads we want
	 * (we actually use 3x as much)
	 */
	
	#pragma omp parallel
	{
		numthreads = omp_get_num_threads();
		//CHUNKING = Total number of columns that will be handled by each thread
		CHUNKING = ((unsigned long)sqrt(MEMSIZE/sizeof(unsigned long))) / numthreads;
		tilesz = (unsigned long)sqrt(MEMSIZE/sizeof(unsigned long)) * sizeof(unsigned long);
		esz = tilesz/sizeof(unsigned long);
	}
	//printf("Sizeof unsigned long: %lu", sizeof(unsigned long));
	printf("The number of threads: %d\nThe chunking is: %lu\nThe tilesz is: %lu\nThat means there are %lu elements per tile\n", numthreads, CHUNKING, tilesz, esz);

	/* initialize all the supporting struct */
	assert(!aml_binding_init(&binding, AML_BINDING_TYPE_SINGLE, 0));
	assert(!aml_tiling_init(&tiling, AML_TILING_TYPE_2D, tilesz, 1, MEMSIZE));
	AML_NODEMASK_ZERO(nodemask);
	AML_NODEMASK_SET(nodemask, 0);
	assert(!aml_arena_jemalloc_init(&arena, AML_ARENA_JEMALLOC_TYPE_REGULAR));

	assert(!aml_area_linux_init(&slow,
				    AML_AREA_LINUX_MANAGER_TYPE_SINGLE,
				    AML_AREA_LINUX_MBIND_TYPE_REGULAR,
				    AML_AREA_LINUX_MMAP_TYPE_ANONYMOUS,
				    &arena, MPOL_BIND, nodemask));
	assert(!aml_area_linux_init(&fast,
				    AML_AREA_LINUX_MANAGER_TYPE_SINGLE,
				    AML_AREA_LINUX_MBIND_TYPE_REGULAR,
				    AML_AREA_LINUX_MMAP_TYPE_ANONYMOUS,
				    &arena, MPOL_BIND, nodemask));
	assert(!aml_dma_linux_seq_init(&dma, numthreads*2));
	assert(!aml_scratch_par_init(&sa, &fast, &slow, &dma, &tiling,
				     2*numthreads, numthreads));
	assert(!aml_scratch_par_init(&sb, &fast, &slow, &dma, &tiling,
				     2*numthreads, numthreads));

	/* allocation */
	a = aml_area_malloc(&slow, MEMSIZE);
	b = aml_area_malloc(&slow, MEMSIZE);
	c = aml_area_malloc(&fast, MEMSIZE);
	assert(a != NULL && b != NULL && c != NULL);

	unsigned long esize = MEMSIZE/sizeof(unsigned long);
	unsigned long numRows = (unsigned long)sqrt(esize);
	for(unsigned long i = 0; i < esize; i++) {
		a[i] = 1;//i % numRows;
		b[i] = 1;//numRows - (i % numRows);
		c[i] = 0;
	}

	/* run kernel */
	#pragma omp parallel for
	for(unsigned long i = 0; i < numthreads; i++) {
		do_work(i);
	}

	/* validate */
	printf("esize = %lu\n", esize);
	int newLines = 0;
	for(unsigned long i = 0; i < esize; i++) {
		printf("%lu ", c[i]);
		newLines++;
		if(newLines == (unsigned long)sqrt(esize)){
			printf("\n");
			newLines = 0;
		}
	}
	printf("\n");

	aml_scratch_par_destroy(&sa);
	aml_scratch_par_destroy(&sb);
	aml_dma_linux_seq_destroy(&dma);
	aml_area_free(&slow, a);
	aml_area_free(&slow, b);
	aml_area_free(&fast, c);
	aml_area_linux_destroy(&slow);
	aml_area_linux_destroy(&fast);
	aml_tiling_destroy(&tiling, AML_TILING_TYPE_1D);
	aml_binding_destroy(&binding, AML_BINDING_TYPE_SINGLE);
	aml_finalize();
	return 0;
}
