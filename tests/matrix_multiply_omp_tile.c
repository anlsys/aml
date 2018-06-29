#include <aml.h>
#include <assert.h>
#include <errno.h>
#include <omp.h>
#include <pthread.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define ITER 10
#define MEMSIZE 33554432//2048//128//67108864//1024 entries by 1024 entries * sizeof(unsigned long)
#define L2_CACHE_SIZE 1048576 //1MB
#define HBM_SIZE 17179869184 //16 GB
#define VERBOSE 1 //Verbose mode will print out extra information about what is happening
#define DEBUG 0 //This will print out verbose messages and debugging statements

size_t numthreads;
//size of 2D Tiles in A matrix
size_t tilesz, esz;
size_t numTiles;
unsigned long CHUNKING;
unsigned long *a, *b, *c;
unsigned long esize, numRows, rowLengthInBytes;
unsigned long rowSizeOfTile, rowSizeInTiles;
unsigned long myId;
AML_TILING_2D_DECL(tiling);
AML_TILING_1D_DECL(tilingB);
AML_AREA_LINUX_DECL(slow);
AML_AREA_LINUX_DECL(fast);
AML_SCRATCH_PAR_DECL(sa);
AML_SCRATCH_PAR_DECL(sb);



int kernel(unsigned long *a, unsigned long *b, unsigned long *c, size_t n, unsigned long colNum, unsigned long tid)
{
	size_t i, j;
	
	for(i = 0; i < rowSizeOfTile; i++)
	{
		for(j = 0; j < rowSizeOfTile; j++)
		{
			c[(unsigned long)(i*rowSizeOfTile + colNum)] += a[(unsigned long)(j + i*rowSizeOfTile)] * b[j];
		}
	}
	return 0;
}


void do_work(unsigned long tid)
{
	myId = tid;
	int offset, i, j, k, ai, bi, oldai, oldbi;
	unsigned long *ap, *bp, *cp;
	void *abaseptr, *bbaseptr;
	offset = tid*CHUNKING;
	
	
	if(DEBUG) printf("Offset tile to begin for thread %lu is: %lu\n", tid, offset);

	ap = aml_tiling_tilestart(&tiling, a, offset);
	bp = aml_tiling_tilestart(&tilingB, b, 0);
	cp = aml_tiling_tilestart(&tiling, c, offset);
	if(DEBUG)printf("Found initial tile starts\n");
	abaseptr = aml_scratch_baseptr(&sa);
	bbaseptr = aml_scratch_baseptr(&sb);
	if(DEBUG)printf("Declared base pointers for a and b\n");
	ai = -1; bi = -1;
	

	//This section works as follows:
	//First loop: This will iterate through CHUNKING tiles within the C matrix
	//Second loop: This will iterate through all the columns of B
	//Third loop: This will iterate through all tiles within A that will contribute to the C matrix
	//Perform the kernel
	
	rowSizeOfTile = aml_tiling_rowsize(&tiling, 0) / sizeof(unsigned long); 
	rowSizeInTiles = numRows / rowSizeOfTile;
	if(DEBUG && tid == 0)printf("The number of rows is: %lu\nThe row size of a tile is %lu\n", numRows, rowSizeOfTile);
	//Iterate through all C tiles
	for(i = 0; i < CHUNKING; i++) {
		if(DEBUG && tid == 0)printf("\n\nBeginning C tile %d of %lu\n", i+1, CHUNKING);
		struct aml_scratch_request *ar, *br;
		unsigned long rowOffsetTile = i / rowSizeInTiles;
		
		//This is equal to number of columns.
		for (j = 0; j < rowSizeOfTile; j++)
		{	
			if(DEBUG && tid == 0)printf("Beginning B column %d of %lu\n", i*rowSizeOfTile + j+1, numRows);
			oldbi = bi;
			bi = !bi;
			aml_scratch_async_pull(&sb, &br, bbaseptr, &bi, b, j+i*rowSizeOfTile);
				
			//This will iterate through the tiles in A that contribute to the respective C tile
			for(k = 0; k < rowSizeInTiles; k++)
			{
				if(DEBUG && tid == 0)printf("Beginning A tile %d of %lu\n", k+1, rowSizeInTiles);
				oldai = ai;
				aml_scratch_async_pull(&sa, &ar, abaseptr, &ai, a, offset+k+1 + rowOffsetTile*rowSizeInTiles);
				kernel(ap, &bp[k*rowSizeOfTile], cp, esz, (unsigned long)j, tid);
				if(DEBUG && tid == 0){
					printf("\n");
					fflush(stdout);
				}
				aml_scratch_wait(&sa, ar);
				ap = aml_tiling_tilestart(&tiling, abaseptr, ai);
				aml_scratch_release(&sa, oldai);
			}
			//for(k = 0; k < rowSizeOfTile; k++){
			//	printf("%lu ", bp[k]);
			//}
			//printf("\n");

			abaseptr = aml_scratch_baseptr(&sa);
			ap = aml_tiling_tilestart(&tiling, a, offset);
			aml_scratch_wait(&sb, br);
			bp = aml_tiling_tilestart(&tilingB, bbaseptr, bi);
			aml_scratch_release(&sb, oldbi);
		}
		//The tile in C should be done and we can now begin the next one
		cp = aml_tiling_tilestart(&tiling, c, offset+i+1);
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
		rowLengthInBytes = (unsigned long)sqrt(MEMSIZE/sizeof(unsigned long)) * sizeof(unsigned long);
		numthreads = omp_get_num_threads();
		//CHUNKING = Total number of columns that will be handled by each thread
		//Tilesz is found by dividing the length (in number of elements) of a dimension by the number of threads.
		//It then checks to see if the tile is too large. If it is, then the size will be reduced until it will fit inside the L2 Cache
		tilesz = ((unsigned long) pow( ( ( (unsigned long)sqrt(MEMSIZE / sizeof(unsigned long)) ) / numthreads), 2) ) * sizeof(unsigned long);
		while(tilesz > L2_CACHE_SIZE/2){
			tilesz = (unsigned long) pow( ( (unsigned long)sqrt(tilesz / sizeof(unsigned long) ) / 2 ), 2);
			if(VERBOSE) printf("Resizing the tile size because it is too large for L2 cache. It is now of size: %lu\n", tilesz);
		}
		numTiles = MEMSIZE / tilesz; 
		CHUNKING = numTiles / numthreads;
		esz = tilesz/sizeof(unsigned long);
		
	}
	
	esize = MEMSIZE/sizeof(unsigned long);
	numRows = (unsigned long)sqrt(esize);
	if(DEBUG)printf("Sizeof unsigned long: %lu", sizeof(unsigned long));
	if(DEBUG || VERBOSE)printf("The total memory size is: %lu\nWe are dealing with a %lu x %lu matrix multiplication\nThe number of threads: %d\nThe chunking is: %lu\nThe tilesz is: %lu\nThat means there are %lu elements per tile\nThere are %lu tiles total\nThe length of a column in bytes is: %lu\n", MEMSIZE, (unsigned long)sqrt(MEMSIZE/sizeof(unsigned long)), (unsigned long)sqrt(MEMSIZE/sizeof(unsigned long)),numthreads, CHUNKING, tilesz, esz, numTiles, rowLengthInBytes);

	/* initialize all the supporting struct */
	assert(!aml_binding_init(&binding, AML_BINDING_TYPE_SINGLE, 0));
	assert(!aml_tiling_init(&tiling, AML_TILING_TYPE_2D, (unsigned long)sqrt(tilesz/sizeof(unsigned long))*sizeof(unsigned long), (unsigned long)sqrt(tilesz/sizeof(unsigned long))*sizeof(unsigned long), tilesz, MEMSIZE));
	assert(!aml_tiling_init(&tilingB, AML_TILING_TYPE_1D, rowLengthInBytes, MEMSIZE));
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
	if(DEBUG)printf("Declaring scratchpad for sa\n");
	assert(!aml_scratch_par_init(&sa, &fast, &slow, &dma, &tiling,
				     2*numthreads, numthreads));
	if(DEBUG)printf("Declaring scratchpad for sb\n");
	assert(!aml_scratch_par_init(&sb, &fast, &slow, &dma, &tilingB,
				     2*numthreads, numthreads));
	if(DEBUG)printf("Sucessfully created both sa and sb\n");
	/* allocation */
	a = aml_area_malloc(&slow, MEMSIZE);
	b = aml_area_malloc(&slow, MEMSIZE);
	c = aml_area_malloc(&fast, MEMSIZE);
	assert(a != NULL && b != NULL && c != NULL);
	if(DEBUG)printf("Allocated space for a, b, and c matrices\n");
	esize = MEMSIZE/sizeof(unsigned long);
	numRows = (unsigned long)sqrt(esize);
	for(unsigned long i = 0; i < esize; i++) {
		a[i] = 1;//i % numRows;
		b[i] = 1;//numRows - (i % numRows);
		c[i] = 0;
	}
	
	int newLines = 0;
	if(DEBUG){
		printf("A MATRIX:\n");
		for(unsigned long i = 0; i < esize; i++) {
			printf("%lu ", a[i]);
			newLines++;
			if(newLines == (unsigned long)sqrt(esize)){
				printf("\n");
				newLines = 0;
			}
		}	
		printf("\nB MATRIX:\n");
	
		for(unsigned long i = 0; i < esize; i++) {
			printf("%lu ", b[i]);
			newLines++;
			if(newLines == (unsigned long)sqrt(esize)){
				printf("\n");
				newLines = 0;
			}
		}
		printf("\n");
	}
	

	/* run kernel */
	#pragma omp parallel for
	for(unsigned long i = 0; i < numthreads; i++) {
		do_work(i);
	}

	/* validate */
	unsigned long correct = 1;
	for(unsigned long i = 0; i < esize; i++){
		if(c[0] != c[i]){
			correct = 0;
		}
	}

	if(DEBUG){
		printf("esize = %lu\n", esize);
		newLines = 0;
		for(unsigned long i = 0; i < esize; i++) {
			printf("%lu ", c[i]);
			newLines++;
			if(newLines == (unsigned long)sqrt(esize)){
				printf("\n");
				newLines = 0;
			}
		}
		printf("\n");
	}

	if(!correct) printf("The matrix multiplication failed. The last incorrect result is at location (%lu, %lu) in the C matrix\n", correct % numRows, correct / numRows);

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
