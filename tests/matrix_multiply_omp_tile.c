#include <aml.h>
#include <assert.h>
#include <errno.h>
#include <omp.h>
#include <pthread.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "mkl.h"
#include <stdlib.h>

#define ITER 10
#define MEMSIZES 134217728//1024 entries by 1024 entries * sizeof(unsigned long)
#define NUMBER_OF_THREADS 32
#define L2_CACHE_SIZE 1048576 //1MB
#define HBM_SIZE 17179869184 //16 GBi
#define VERBOSE 1 //Verbose mode will print out extra information about what is happening
#define DEBUG 1 //This will print out verbose messages and debugging statements
#define PRINT_ARRAYS 1 
#define HBM 1 

size_t numthreads;
//size of 2D Tiles in A matrix
size_t tilesz, esz;
size_t numTiles;
unsigned long CHUNKING;
double *a, *b, *c;
unsigned long MEMSIZE;
unsigned long esize, numRows, rowLengthInBytes;
unsigned long rowSizeOfTile, rowSizeInTiles;
unsigned long myId;
unsigned long long beginTime, endTime;
unsigned long long waitingTime, totalWait;
clock_t startClock, endClock;
AML_TILING_2D_DECL(tiling);
AML_TILING_2D_DECL(tilingB);
AML_AREA_LINUX_DECL(slow);
AML_AREA_LINUX_DECL(fast);
AML_SCRATCH_PAR_DECL(sa);
AML_SCRATCH_PAR_DECL(sb);

//This code will take cycles executed as a use for timing the kernel.
uint64_t rdtsc(){
    unsigned int lo,hi;
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return ((uint64_t)hi << 32) | lo;
}

//This temporary kernel will be our dgemm replacement.
void multiplyTiles(double *a, double *b, double *c, int n, int m){
	unsigned long i, j, k;
	
	for(i = 0; i < m; i++){
		for(j = 0; j < n; j++){
			for(k = 0; k < m; k++){
				c[i*m + j] += a[i*m + k] * b[j + k*n]; 
			}		
		}
	}

}

void do_work()
{
	if(DEBUG) printf("Inside do_work()\n");	
	int offset, i, j, k, l, ai, bi, iMod, oldai, oldbi, tilesPerCol;
	double *ap, *bp, *cp, *apLoc, *bpLoc;
	void *abaseptr, *bbaseptr;
	        
	if(HBM){
		abaseptr = aml_scratch_baseptr(&sa);
		bbaseptr = aml_scratch_baseptr(&sb);
		if(DEBUG) printf("Got a and b baseptrs\n");
	}
	ai = -1; bi = -1;
	
	ap = a;
	bp = b;
	cp = c;

	//This section works as follows:
	//OUTER LOOP: Begin by requesting the next column of A tiles
	//INNER LOOP: Request the next columns of B tiles
	//Fork off and begin working on current tiles
	//Run kernel and compute partial multiplications
	//End forking
	//Wait on B tiles, then repeat INNER LOOP
	//Wait on A tiles, then repeat OUTER LOOP
	//Mult done
	struct aml_scratch_request *ar, *br;

	tilesPerCol = rowSizeInTiles / numthreads; //This should evaluate to an integer value
		
	//This will iterate through each column of tiles in A matrix
	//This loop is O(rowSizeInTiles) (O(n))
	if(DEBUG) printf("tilesPerCol = %d\n Beginning outerloop\n", tilesPerCol);
	for(i = 0; i < rowSizeInTiles; i++) {
		
		//Request next column of tiles from A into the scratchpad for A	
		if(HBM){
			oldbi = bi;
			oldai = ai;
			aml_scratch_async_pull(&sa, &ar, abaseptr, &ai, a, i + 1);
		 	aml_scratch_async_pull(&sb, &br, bbaseptr, &bi, b, i + 1);
			if(DEBUG) printf("Async pulling next columns of B and A (%d)\n", i);	
		}		
		//This will begin the dispersion of work accross threads, this loop is actually O(1) 
		#pragma omp parallel for
		for(k = 0; k < numthreads; k++)
		{
			//This loop will cover if threads are handling multiple rows of tiles. This shouldn't be a necessarily large number
			//This loop is technically O(n) but in reality will usually be O(1) because tilesPerCol will be a small number relative to rowSizeInTiles
			for(j = 0; j < tilesPerCol; j++){
				offset = (k * tilesPerCol) + j;
				if(DEBUG) printf("Thread %d has an offset value of %d\n", k, offset);
				//This will give the beginning offset for where each thread should point to in the tilingB sized ap array
				apLoc = aml_tiling_tilestart(&tiling, ap, offset);
				offset = (k * tilesPerCol) + j;
						
				//Now we will iterate through all the tiles in the row of B tiles and compute a partial matrix multiplication
				//This loop is O(n)
				for(l = 0; l < rowSizeInTiles; l++){
					
					bpLoc = aml_tiling_tilestart(&tiling, bp, l);
					
					//This will begin at the tile row that is at x cordinate equal to bp offset, and y cordinate equal to ap offset
					cp = aml_tiling_tilestart(&tiling, c, ((int)rowSizeInTiles * ((k * tilesPerCol) + j)) + l);
					if(DEBUG) printf("Thread %d is beginning tile c at tile offset %d (offset = %d)\n", k, (int)rowSizeInTiles * ((k * tilesPerCol) + j) + l, offset);
					
					if(0 && DEBUG && k == 0){
						printf("Printing off the matrix tiles being worked on\n A MATRIX:\n");
						int tempI, tempJ;
						for(tempI = 0; tempI < rowSizeOfTile; tempI++){
							for(tempJ = 0; tempJ < rowSizeOfTile; tempJ++){
								printf("%lf ", apLoc[tempI*rowSizeOfTile + tempJ]);
							}
							printf("\n");
						} 
						printf("\nB MATRIX:\n");
						for(tempI = 0; tempI < rowSizeOfTile; tempI++){
							for(tempJ = 0; tempJ < rowSizeOfTile; tempJ++){
								printf("%lf ", bpLoc[tempI*rowSizeOfTile + tempJ]);
							}
							printf("\n");
						} 

					}
					if(DEBUG && k == 0) printf("Beginning matrix multiply\n");
					//Currently we will call the user written kernel, but we will eventually use dgemm???
					multiplyTiles(apLoc, bpLoc, cp, rowSizeOfTile, rowSizeOfTile);
					
					//cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rowSizeOfTile, rowSizeOfTile, rowSizeOfTile, 1.0, apLoc, rowSizeOfTile, bpLoc, rowSizeOfTile, 1.0, cpLoc, rowSizeOfTile);
					//if(DEBUG && k == 0) printf("Returned from matrix multiply\n");
					if(0 && DEBUG && k == 0){
						int tempI, tempJ;
						printf("\nC MATRIX:\n");
						for(tempI = 0; tempI < rowSizeOfTile; tempI++){
							for(tempJ = 0; tempJ < rowSizeOfTile; tempJ++){
								printf("%lf ", cp[tempI*rowSizeOfTile + tempJ]);
							}
							printf("\n");
						} 

					}
				}			
			}
			
		}	
		if(DEBUG){
			printf("Current C matrix\n");
			int newLines = 0;
			for(unsigned long i = 0; i < esize; i++) {
				printf("%lf ", cp[i]);
				newLines++;
				if(newLines == (unsigned long)sqrt(esize)){
					printf("\n");
					newLines = 0;
				}
			}
			printf("\n");
		}

		
		if(HBM){ 
			aml_scratch_wait(&sa, ar);
			aml_scratch_wait(&sb, br);
			
			ap = aml_tiling_tilestart(&tilingB, abaseptr, ai);
			bp = aml_tiling_tilestart(&tilingB, bbaseptr, bi);
			aml_scratch_release(&sa, oldai);
			aml_scratch_release(&sb, oldbi);
		}
		else{
			ap = aml_tiling_tilestart(&tilingB, a, i + 1);
			bp = aml_tiling_tilestart(&tilingB, b, i + 1);
		}		
	
	}
	
}


//This matrix multiplication will implement matrix multiplication in the following way:
//	The A, B, and C matrices will be broken into tiles that edge as close to 1 MB as possible (size of L2 cache) while also allowing all threads to work on data.
//	The algorithm will chunk up the work dependent on number of tiles. The multiplication will go as follows:
//		1) The algorithm will take a column of the tiles of A. (Prefetch next column of A tiles to fast memory).
//		2) The algorithm will take a column of B tiles (prefetch next column into fast memory).
//		3) Perform partial matrix multiplications using A tile and B Tile (using dgemm hopefully). 
//		4) Repeat 2 & 3 until columns of B tiles are exhausted. Then continue
//		5) Repeat from 1 until columns of A tiles are exhausted. Then continue
//		6) DONE
//Another potential solution could be to tile the B matrix as well. This will require Atomic Additions though.  
int main(int argc, char *argv[])
{
	AML_BINDING_SINGLE_DECL(binding);
	AML_ARENA_JEMALLOC_DECL(arena);
	AML_DMA_LINUX_SEQ_DECL(dma);
	unsigned long nodemask[AML_NODEMASK_SZ];
	aml_init(&argc, &argv);
	if(argc == 1){
		if(VERBOSE) printf("No arguments provided, setting numThreads = %d and Memsize = %lu\n", NUMBER_OF_THREADS, MEMSIZE);
		omp_set_num_threads(NUMBER_OF_THREADS);
		MEMSIZE = MEMSIZES;
	}
	else if(argc == 2){	
		if(VERBOSE) printf("Setting number of threads\n");
		omp_set_num_threads(NUMBER_OF_THREADS);
	 	if(VERBOSE) printf("Setting MEMSIZE\n");	
		MEMSIZE = atol(argv[1]);
		if(VERBOSE) printf("1 argument provided, setting numThreads = %d and Memsize = %lu\n", NUMBER_OF_THREADS, MEMSIZE);
	}
	else if(argc >= 3){
		omp_set_num_threads(atoi(argv[2]));
		MEMSIZE = atol(argv[1]);
		if(VERBOSE) printf("Two arguments provided, setting numThreads = %d and Memsize = %lu\n", atoi(argv[2]), atol(argv[1]));

	}
	
	/* use openmp env to figure out how many threads we want
	 * (we actually use 3x as much)
	 */
	esize = MEMSIZE/sizeof(unsigned long);
	numRows = (unsigned long)sqrt(esize);

	#pragma omp parallel
	{
		rowLengthInBytes = (unsigned long)sqrt(MEMSIZE/sizeof(double)) * sizeof(double);
		numthreads = omp_get_num_threads();
		//CHUNKING = Total number of columns that will be handled by each thread
		//Tilesz is found by dividing the length (in number of elements) of a dimension by the number of threads.
		//It then checks to see if the tile is too large. If it is, then the size will be reduced until it will fit inside the L2 Cache
		tilesz = ((unsigned long) pow( ( ( sqrt(MEMSIZE / sizeof(double)) ) / numthreads), 2) ) * sizeof(double);
		double multiplier = 2;
		while(tilesz > L2_CACHE_SIZE && argc != 4){
			tilesz = pow(sqrt(MEMSIZE/sizeof(double)) / (numthreads*multiplier),2)*sizeof(double);
			
			if(VERBOSE) printf("Resizing the tile size because it is too large for L2 cache. It is now of size: %lu\n", tilesz);
		}
		if(argc == 4){
			tilesz = atol(argv[3]);
		}
		numTiles = MEMSIZE / tilesz; 
		CHUNKING = numTiles / numthreads;
		esz = tilesz/sizeof(double);
		
	}

		
	if(DEBUG) printf("Sizeof double: %d\n", sizeof(double));
	if(DEBUG || VERBOSE)printf("The total memory size is: %lu\nWe are dealing with a %lu x %lu matrix multiplication\nThe number of threads: %d\nThe chunking is: %lu\nThe tilesz is: %lu\nThat means there are %lu elements per tile\nThere are %lu tiles total\nThe length of a column in bytes is: %lu\n", MEMSIZE, (unsigned long)sqrt(MEMSIZE/sizeof(unsigned long)), (unsigned long)sqrt(MEMSIZE/sizeof(unsigned long)),numthreads, CHUNKING, tilesz, esz, numTiles, rowLengthInBytes);

	/* initialize all the supporting struct */
	assert(!aml_binding_init(&binding, AML_BINDING_TYPE_SINGLE, 0));
	assert(!aml_tiling_init(&tiling, AML_TILING_TYPE_2D, (unsigned long)sqrt(tilesz/sizeof(double))*sizeof(double), (unsigned long)sqrt(tilesz/sizeof(double))*sizeof(double), tilesz, MEMSIZE));	
	rowSizeOfTile = aml_tiling_rowsize(&tiling, 0) / sizeof(double); 
	rowSizeInTiles = numRows / rowSizeOfTile;
	//This tiling B will be used for scratch pad memory movement of large tiled columns
	assert(!aml_tiling_init(&tilingB, AML_TILING_TYPE_2D, rowSizeOfTile * sizeof(double), rowSizeInTiles * rowSizeOfTile * sizeof(double), tilesz * rowSizeInTiles, MEMSIZE));
	if(DEBUG) printf("Row size in tiles = %lu\n", rowSizeInTiles);
	AML_NODEMASK_ZERO(nodemask);
	AML_NODEMASK_SET(nodemask, 0);
	assert(!aml_arena_jemalloc_init(&arena, AML_ARENA_JEMALLOC_TYPE_REGULAR));

	assert(!aml_area_linux_init(&slow,
				    AML_AREA_LINUX_MANAGER_TYPE_SINGLE,
				    AML_AREA_LINUX_MBIND_TYPE_REGULAR,
				    AML_AREA_LINUX_MMAP_TYPE_ANONYMOUS,
				    &arena, MPOL_BIND, nodemask));
	AML_NODEMASK_ZERO(nodemask);
	AML_NODEMASK_SET(nodemask, 1);
	assert(!aml_area_linux_init(&fast,
				    AML_AREA_LINUX_MANAGER_TYPE_SINGLE,
				    AML_AREA_LINUX_MBIND_TYPE_REGULAR,
				    AML_AREA_LINUX_MMAP_TYPE_ANONYMOUS,
				    &arena, MPOL_BIND, nodemask));
	
	assert(!aml_dma_linux_seq_init(&dma, numthreads*2));
	if(HBM){
		if(DEBUG)printf("Declaring scratchpad for sa\n");
		assert(!aml_scratch_par_init(&sa, &fast, &slow, &dma, &tilingB,
				     2*numthreads, numthreads));
		if(DEBUG)printf("Declaring scratchpad for sb\n");
		assert(!aml_scratch_par_init(&sb, &fast, &slow, &dma, &tilingB,
				     2*numthreads, numthreads));
		if(DEBUG)printf("Sucessfully created both sa and sb\n");
	}
	/* allocation */
	a = aml_area_malloc(&slow, MEMSIZE);
	b = aml_area_malloc(&slow, MEMSIZE);
	if(HBM){
		c = aml_area_malloc(&fast, MEMSIZE);
	}
	else{
		c = aml_area_malloc(&slow, MEMSIZE);
	}
	assert(a != NULL && b != NULL && c != NULL);
	if(DEBUG)printf("Allocated space for a, b, and c matrices\n");
	esize = MEMSIZE/sizeof(double);
	numRows = (unsigned long)sqrt(esize);
	for(unsigned long i = 0; i < esize; i++) {
		a[i] = 1.0;//i % numRows;
		b[i] = 1.0;//numRows - (i % numRows);
		c[i] = 0.0;
	}
	if(VERBOSE) printf("esize = %lu\n", esize);
	
	int newLines = 0;
	if(PRINT_ARRAYS){
		printf("A MATRIX:\n");
		for(unsigned long i = 0; i < esize; i++) {
			printf("%lf ", a[i]);
			newLines++;
			if(newLines == (unsigned long)sqrt(esize)){
				printf("\n");
				newLines = 0;
			}
		}	
		printf("\nB MATRIX:\n");
	
		for(unsigned long i = 0; i < esize; i++) {
			printf("%lf ", b[i]);
			newLines++;
			if(newLines == (unsigned long)sqrt(esize)){
				printf("\n");
				newLines = 0;
			}
		}
		printf("\n");
	}
	

	/* run kernel */
	startClock = clock();
	beginTime = rdtsc();

	//BEGIN MULTIPLICATION
//	#pragma omp parallel for
//	for(unsigned long i = 0; i < numthreads; i++){
		do_work();
//	}
	//END MULTIPLICATION
	endTime = rdtsc();
	endClock = clock();
	printf("Kernel Timing Statistics:\nRDTSC: %lu cycles\nCLOCK: %lf Seconds\nCycles waiting: %lu\n", endTime - beginTime, (double)(endClock - startClock) / CLOCKS_PER_SEC, totalWait);

	/* validate */
	unsigned long correct = 1;
	for(unsigned long i = 0; i < esize; i++){
		if(c[0] != c[i]){
			correct = 0;
		}
	}

	if(PRINT_ARRAYS){
		printf("esize = %lu\n", esize);
		newLines = 0;
		for(unsigned long i = 0; i < esize; i++) {
			printf("%lf ", c[i]);
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
	if(HBM) aml_area_free(&fast, c);
	if(!HBM) aml_area_free(&slow, c);
	aml_area_linux_destroy(&slow);
	aml_area_linux_destroy(&fast);
	aml_tiling_destroy(&tiling, AML_TILING_TYPE_1D);
	aml_binding_destroy(&binding, AML_BINDING_TYPE_SINGLE);
	aml_finalize();
	return 0;
}
