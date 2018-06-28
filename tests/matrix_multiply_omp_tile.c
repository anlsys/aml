#include <aml.h>
#include <assert.h>
#include <errno.h>
#include <omp.h>
#include <pthread.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define ITER 10
#define MEMSIZE 2048//67108864//1024 entries by 1024 entries * sizeof(unsigned long)
#define L2_CACHE_SIZE 1048576 //1MB
#define HBM_SIZE 17179869184 //16 GB


size_t numthreads;
//size of 2D Tiles in A matrix
size_t tilesz, esz;
size_t numTiles;
unsigned long CHUNKING;
unsigned long *a, *b, *c;
unsigned long esize, numRows;
unsigned long rowSizeOfTile, rowSizeInTiles;
AML_TILING_2D_DECL(tiling);
AML_TILING_1D_DECL(tilingB);
AML_AREA_LINUX_DECL(slow);
AML_AREA_LINUX_DECL(fast);
AML_SCRATCH_PAR_DECL(sa);
AML_SCRATCH_PAR_DECL(sb);



int kernel(unsigned long *a, unsigned long *b, unsigned long *c, size_t n)
{
	size_t i, j;
	
	for(i = 0; i < rowSizeOfTile; i++)
	{
		for(j = 0; j < rowSizeOfTile; j++)
		{
			c[i*rowSizeOfTile] += a[j + i*rowSizeOfTile] * b[j];
		}
	}
	return 0;
}


void do_work(unsigned long tid)
{
	int offset, i, j, k, ai, bi, oldai, oldbi;
	unsigned long *ap, *bp, *cp;
	void *abaseptr, *bbaseptr;
	offset = tid*CHUNKING;
	
	
	ap = aml_tiling_tilestart(&tiling, a, offset);
	bp = aml_tiling_tilestart(&tilingB, b, 0);
	cp = aml_tiling_tilestart(&tiling, c, offset);
	//printf("Found initial tile starts\n");
	abaseptr = aml_scratch_baseptr(&sa);
	bbaseptr = aml_scratch_baseptr(&sb);
	//printf("Declared base pointers for a and b\n");
	ai = -1; bi = -1;
	

	//This section works as follows:
	//First loop: This will iterate through CHUNKING tiles within the C matrix
	//Second loop: This will iterate through all the columns of B
	//Third loop: This will iterate through all tiles within A that will contribute to the C matrix
	//Perform the kernel
	
	rowSizeOfTile = aml_tiling_rowsize(&tiling, 0) / sizeof(unsigned long); 
	rowSizeInTiles = numRows / rowSizeOfTile;
	printf("The number of rows is: %lu\n", numRows);
	//Iterate through all C tiles
	for(i = 0; i < CHUNKING; i++) {
		struct aml_scratch_request *ar, *br;
		//printf("Declared scratch requests\n");
		unsigned long rowOffsetTile = i / rowSizeInTiles;
		
		//This is equal to number of columns.
		for (j = 0; j < numRows; j++)
		{
			oldbi = bi;
			if(j == esz - 1){
				aml_scratch_async_pull(&sb, &br, bbaseptr, &bi, b, 0);
			}
			else{
				printf("Attempting async pull for b: %lu\n", tid);
				aml_scratch_async_pull(&sb, &br, bbaseptr, &bi, b, j+1);
				printf("Sucessfully requested async pull for b: %lu\n", tid);
			}
			//This will iterate through the tiles in A that contribute to the respective C tile
			for(k = 0; k < rowSizeInTiles; k++)
			{
				oldai = ai;
				if(k != rowSizeInTiles - 1){
					printf("Attempting async pull on a: %d, %lu\n", k, tid);
					aml_scratch_async_pull(&sa, &ar, abaseptr, &ai, a, offset+k+1 + rowOffsetTile*rowSizeInTiles);
					printf("Sucessfully requested async pull on a: %lu\n", tid);
				}
				kernel(ap, bp + k*rowSizeOfTile, cp + k, esz);
				if(k != rowSizeInTiles - 1){
					printf("Waiting for async pull for a to finish: %lu\n", tid);
					aml_scratch_wait(&sa, ar);
					printf("Completed async pull for a: %lu\n", tid);
				}
				ap = aml_tiling_tilestart(&tiling, abaseptr, k);
	
			}
			abaseptr = aml_scratch_baseptr(&sa);
			printf("Waiting for async pull on b to finish: %lu\n", tid);	
			aml_scratch_wait(&sb, br);
			printf("The async pull on b is done and we can continue: %lu\n", tid);
			bp = aml_tiling_tilestart(&tilingB, bbaseptr, j);
			aml_scratch_release(&sb, oldbi);
		}
		//The tile in C should be done and we can now begin the next one
		cp = aml_tiling_tilestart(&tiling, c, offset+i+1);
		aml_scratch_release(&sa, oldai);
	}
	//Third argument may be wrong
	for (j = 0; j < esz; j++){
		kernel(ap, bp, cp + j, esz);	
	}
	

}


//This matrix multiplication will implement matrix multiplication in the following way:
//	The A matrix will be broken into tiles that edge as close to 512 KB as possible (half the size of L2 cache).
//	The B matrix will be broken into columns (Placed in high bandwidth memory (HBM)) Also the matrix is assummed to be transposed
//	The C matrix will be broken into tiles equal in size to the A matrix tiles (Will exist in HBM, but is other half of L2 cache)
//	The algorithm will chunk up the work dependent on number of tiles. The multiplication will go as follows:
//		Tile from A will multiply all of its rows by the respective partial columns of B.
//		The results will be placed in C tile at the X position of the B Column and Y Position of the A tile row
//		This will iterate through all A tiles in the chunk. Upon finishing, it will move to the next B Column
//		Upon iterating through all B columns, the C matrix will be complete.
//Another potential solution could be to tile the B matrix as well. This will require Atomic Additions though.  
int main(int argc, char *argv[])
{
	AML_BINDING_SINGLE_DECL(binding);
	AML_ARENA_JEMALLOC_DECL(arena);
	AML_DMA_LINUX_SEQ_DECL(dma);
	unsigned long nodemask[AML_NODEMASK_SZ];
	aml_init(&argc, &argv);
	assert(argc == 1);

	omp_set_num_threads(1);
	/* use openmp env to figure out how many threads we want
	 * (we actually use 3x as much)
	 */
	
	#pragma omp parallel
	{
		numthreads = omp_get_num_threads();
		//CHUNKING = Total number of columns that will be handled by each thread
		//Tilesz is found by dividing the length (in number of elements) of a dimension by the number of threads.
		//It then checks to see if the tile is too large. If it is, then the size will be reduced until it will fit inside the L2 Cache
		tilesz = ((unsigned long) pow( ( ( (unsigned long)sqrt(MEMSIZE / sizeof(unsigned long)) ) / numthreads), 2) ) * sizeof(unsigned long);
		while(tilesz > L2_CACHE_SIZE/2){
			tilesz = (unsigned long) pow( ( (unsigned long)sqrt(tilesz / sizeof(unsigned long) ) / 2 ), 2);
		}
		numTiles = MEMSIZE / tilesz; 
		CHUNKING = numTiles / numthreads;
		esz = tilesz/sizeof(unsigned long);
		
	}
	esize = MEMSIZE/sizeof(unsigned long);
	numRows = (unsigned long)sqrt(esize);
	//printf("Sizeof unsigned long: %lu", sizeof(unsigned long));
	printf("The total memory size is: %lu\nWe are dealing with a %lu x %lu matrix multiplication\nThe number of threads: %d\nThe chunking is: %lu\nThe tilesz is: %lu\nThat means there are %lu elements per tile\nThere are %lu tiles total\n", MEMSIZE, (unsigned long)sqrt(MEMSIZE/sizeof(unsigned long)), (unsigned long)sqrt(MEMSIZE/sizeof(unsigned long)),numthreads, CHUNKING, tilesz, esz, numTiles);

	/* initialize all the supporting struct */
	assert(!aml_binding_init(&binding, AML_BINDING_TYPE_SINGLE, 0));
	assert(!aml_tiling_init(&tiling, AML_TILING_TYPE_2D, (unsigned long)sqrt(tilesz/sizeof(unsigned long))*sizeof(unsigned long), (unsigned long)sqrt(tilesz/sizeof(unsigned long))*sizeof(unsigned long), tilesz, MEMSIZE));
	assert(!aml_tiling_init(&tilingB, AML_TILING_TYPE_1D, tilesz, MEMSIZE));
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
	//printf("Declaring scratchpad for sa\n");
	assert(!aml_scratch_par_init(&sa, &fast, &slow, &dma, &tiling,
				     2*numthreads, numthreads));
	//printf("Declaring scratchpad for sb\n");
	assert(!aml_scratch_par_init(&sb, &fast, &slow, &dma, &tilingB,
				     2*numthreads, numthreads));
	//printf("Sucessfully created both sa and sb\n");
	/* allocation */
	a = aml_area_malloc(&slow, MEMSIZE);
	b = aml_area_malloc(&slow, MEMSIZE);
	c = aml_area_malloc(&fast, MEMSIZE);
	assert(a != NULL && b != NULL && c != NULL);
	//printf("Allocated space for a, b, and c matrices\n");
	esize = MEMSIZE/sizeof(unsigned long);
	numRows = (unsigned long)sqrt(esize);
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
