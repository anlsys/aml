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
#define HBM_SIZE 17179869184 //16 GB

#define VERBOSE 0 //Verbose mode will print out extra information about what is happening
#define DEBUG 0 //This will print out verbose messages and debugging statements
#define PRINT_ARRAYS 0 
#define BILLION 1000000000L

#include <aml.h>

AML_TILING_2D_DECL(tiling);
AML_TILING_2D_DECL(tilingB);
AML_AREA_LINUX_DECL(slow);
AML_SCRATCH_PAR_DECL(sa);
AML_SCRATCH_PAR_DECL(sb);


size_t numthreads;
//size of 2D Tiles in A matrix
size_t tilesz, esz;
size_t numTiles;
unsigned long CHUNKING;
double *a, *b, *c;
unsigned long MEMSIZE;
unsigned long esize, numRows, rowLengthInBytes;
unsigned long rowSizeOfTile, rowSizeInTiles;
unsigned long long beginTime, endTime;
struct timespec startClock, endClock;
double elapsedTime;



//This code will take cycles executed as a use for timing the kernel.

unsigned long rdtsc(){
    unsigned int lo,hi;
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return ((unsigned long)hi << 32) | lo;
}


void do_work()
{
	int i, k, ai, bi, oldai, oldbi, tilesPerCol;
	MKL_INT lda = (int)rowSizeOfTile, ldb, ldc;
	ldb = lda;
	ldc = lda;
	double *ap, *bp, *cp;
	void *abaseptr, *bbaseptr;
	        
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
	for(i = 0; i < rowSizeInTiles; i++) {	
		//This will begin the dispersion of work accross threads, this loop is actually O(1) 
		#pragma omp parallel for
		for(k = 0; k < numthreads; k++)
		{
			int j, l, offset;
			double *apLoc, *bpLoc, *cpLoc;
			//This loop will cover if threads are handling multiple rows of tiles. This shouldn't be a necessarily large number
			//This loop is technically O(n) but in reality will usually be O(1) because tilesPerCol will be a small number relative to rowSizeInTiles
			for(j = 0; j < tilesPerCol; j++){
				offset = (k * tilesPerCol) + j;
				//This will give the beginning offset for where each thread should point to in the tilingB sized ap array
				apLoc = aml_tiling_tilestart(&tiling, ap, offset);
				offset = (k * tilesPerCol) + j;		
				//Now we will iterate through all the tiles in the row of B tiles and compute a partial matrix multiplication
				//This loop is O(n)
				for(l = 0; l < rowSizeInTiles; l++){
					bpLoc = aml_tiling_tilestart(&tiling, bp, l);
					//This will begin at the tile row that is at x cordinate equal to bp offset, and y cordinate equal to ap offset
					offset = (((int)rowSizeInTiles * ((k * tilesPerCol)+ j) ) + l);
					cp = aml_tiling_tilestart(&tiling, c, offset);

					cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ldc, lda, ldb, 1.0, apLoc, lda, bpLoc, ldb, 1.0, cp, ldc);
				}			
			}
			
		}	

		ap = aml_tiling_tilestart(&tilingB, a, i + 1);
		bp = aml_tiling_tilestart(&tilingB, b, i + 1);	
	
	}
	
}

int argoMM(int argc, char* argv[]){
		AML_BINDING_SINGLE_DECL(binding);
		AML_ARENA_JEMALLOC_DECL(arena);
		unsigned long nodemask[AML_NODEMASK_SZ];
		aml_init(&argc, &argv);
		esize = (unsigned long)MEMSIZE/sizeof(double);
		numRows = (unsigned long)sqrt(esize);
	
		#pragma omp parallel
		{
			rowLengthInBytes = (unsigned long)sqrt(MEMSIZE/sizeof(double)) * sizeof(double);
			numthreads = omp_get_num_threads();
			tilesz = ((unsigned long) pow( ( ( sqrt(MEMSIZE / sizeof(double)) ) / (numthreads * 2) ), 2) ) * sizeof(double);
			double multiplier = 2;
			if(argc == 4){
				tilesz = atol(argv[3]);
			}
			numTiles = MEMSIZE / tilesz; 
			CHUNKING = numTiles / numthreads;
			esz = tilesz/sizeof(double);
		}
	
			
		if(DEBUG || VERBOSE)printf("The total memory size is: %lu\nWe are dealing with a %lu x %lu matrix multiplication\nThe number of threads: %d\nThe chunking is: %lu\nThe tilesz is: %lu\nThat means there are %lu elements per tile\nThere are %lu tiles total\nThe length of a column in bytes is: %lu\n", MEMSIZE, (unsigned long)sqrt(MEMSIZE/sizeof(unsigned long)), (unsigned long)sqrt(MEMSIZE/sizeof(unsigned long)),numthreads, CHUNKING, tilesz, esz, numTiles, rowLengthInBytes);

		assert(!aml_binding_init(&binding, AML_BINDING_TYPE_SINGLE, 0));
		assert(!aml_tiling_init(&tiling, AML_TILING_TYPE_2D, (unsigned long)sqrt(tilesz/sizeof(double))*sizeof(double), (unsigned long)sqrt(tilesz/sizeof(double))*sizeof(double), tilesz, MEMSIZE));	
		rowSizeOfTile = aml_tiling_tilerowsize(&tiling, 0) / sizeof(double); 
		rowSizeInTiles = numRows / rowSizeOfTile;
		assert(!aml_tiling_init(&tilingB, AML_TILING_TYPE_2D, rowSizeOfTile * sizeof(double), rowSizeInTiles * rowSizeOfTile * sizeof(double), tilesz * rowSizeInTiles, MEMSIZE));
		AML_NODEMASK_ZERO(nodemask);
		AML_NODEMASK_SET(nodemask, 0);
		assert(!aml_arena_jemalloc_init(&arena, AML_ARENA_JEMALLOC_TYPE_REGULAR));
		assert(!aml_area_linux_init(&slow,
					    AML_AREA_LINUX_MANAGER_TYPE_SINGLE,
					    AML_AREA_LINUX_MBIND_TYPE_REGULAR,
					    AML_AREA_LINUX_MMAP_TYPE_ANONYMOUS,
					    &arena, MPOL_BIND, nodemask));
		
		/* allocation */
		a = aml_area_malloc(&slow, MEMSIZE);
		b = aml_area_malloc(&slow, MEMSIZE);
		c = aml_area_malloc(&slow, MEMSIZE);
		
		assert(a != NULL && b != NULL && c != NULL);
		esize = MEMSIZE/sizeof(double);
		numRows = (unsigned long)sqrt(esize);
		for(unsigned long i = 0; i < esize; i++) {
			a[i] = 1.0;//i % numRows;
			b[i] = 1.0;//numRows - (i % numRows);
			c[i] = 0.0;
		}	
		int newLines = 0;	
		//This will execute on core 0
		clock_gettime(CLOCK_REALTIME, &startClock);
		beginTime = rdtsc();
	
		do_work();
	
		//This will execute on core 0
		endTime = rdtsc();
		clock_gettime(CLOCK_REALTIME, &endClock);

		elapsedTime = BILLION * ( endClock.tv_sec - startClock.tv_sec ) + (( endClock.tv_nsec - startClock.tv_nsec ));
		printf("%lu\t%lf\n", endTime - beginTime, elapsedTime);
	
		/* validate */
		unsigned long correct = 1;
		for(unsigned long i = 0; i < esize; i++){
			if(c[0] != c[i]){
				correct = 0;
			}
		}
	
		if(!correct){
			printf("The matrix multiplication failed. The last incorrect result is at location C(0,0) = %lf in the C matrix\n", c[0]);
		}	
	
}


//This matrix multiplication will implement matrix multiplication in the following way:
//	The A, B, and C matrices will be broken into tiles that edge as close to 1 MB as possible (size of L2 cache) while also allowing all threads to work on data.
//	The algorithm will chunk up the work dependent on number of tiles. The multiplication will go as follows:
//		1) The algorithm will take a column of the tiles of A. (Prefetch next column of A tiles to fast memory).
//		2) The algorithm will take a column of B tiles (prefetch next column into fast memory).
//		3) Perform partial matrix multiplications using A tile and B Tile (using dgemm). 
//		4) Repeat partial matrix multiplications until A and B columns are exhausted.
//		5) DONE
//Another potential solution could be to tile the B matrix as well. This will require Atomic Additions though.  
int main(int argc, char *argv[])
{
	if(argc == 1){
		if(VERBOSE) printf("No arguments provided, setting numThreads = %d and Memsize = %lu\n", NUMBER_OF_THREADS, MEMSIZE);
		omp_set_num_threads(NUMBER_OF_THREADS);
		MEMSIZE = MEMSIZES;
	}
	else if(argc == 2){	
		if(VERBOSE) printf("Setting number of threads\n");
		omp_set_num_threads(NUMBER_OF_THREADS);
	 	if(VERBOSE) printf("Setting MEMSIZE\n");	
		MEMSIZE = sizeof(double)*(atol(argv[1]) * atol(argv[1]));
		if(VERBOSE) printf("1 argument provided, setting numThreads = %d and Memsize = %lu\n", NUMBER_OF_THREADS, MEMSIZE);
	}
	else if(argc >= 3){
		omp_set_num_threads(atoi(argv[2]));
		MEMSIZE = sizeof(double)*(atol(argv[1]) * atol(argv[1]));
		if(VERBOSE) printf("Two arguments provided, setting numThreads = %d and Memsize = %lu\n", atoi(argv[2]), atol(argv[1]));
	}

	argoMM(argc, argv);

	return 0;
}
