#include <assert.h>
#include <errno.h>
#include <mkl.h>
#include <omp.h>
#include <pthread.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
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
AML_AREA_LINUX_DECL(fast);
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




//No Matrix is transposed for this algorithm anymore; however, a tile is assummed to be contiguous in memory.
//This algorithm will work as follows:
//1) Prefetch top row of tiles from A, B, and C
//2) All Threads will begin on A[0]th tile of that row and do the following:
//3) Grab the tiles within the first row of B and C that correspond to their threadnum.
//4) Perform a matrix multiplication of their tiles. Once all the threads finish, the next row of B tiles is grabbed and the next tile in A is used.
//5) Repeat until no more B rows / no A tiles
//6) The C row of tiles is complete and the next A and C rows can be acquired.
//7) Repeat process until no more A and C rows. 
//Matrix mutliplication is now complete.
void do_work()
{
	int i, tilesPerCol;
	int lda = (int)rowSizeOfTile, ldb, ldc;
	ldb = lda;
	ldc = lda;
	double *ap, *bp, *cp;
	int colSizeInTiles = rowSizeInTiles;
	       
	ap = a;
	bp = b;
	cp = c;	
	//This is performing 7) in the comments, it will end when A and C tiles are all done		
	for(i = 0; i < colSizeInTiles; i++) {
		//Request next column of tiles from A into the scratchpad for A	
		int l, k, offset;
		//This loop will go through each row of B and each tile in a row of A
		for(k = 0; k < rowSizeInTiles; k++){
			double *apLoc;
			apLoc = aml_tiling_tilestart(&tiling, ap, k);
			#pragma omp parallel for
			for(l = 0; l < rowSizeInTiles; l++){
				double *bpLoc, *cpLoc;
				cpLoc = aml_tiling_tilestart(&tiling, cp, l);			
				bpLoc = aml_tiling_tilestart(&tiling, bp, l);
				cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ldc, lda, ldb, 1.0, apLoc, lda, bpLoc, ldb, 1.0, cpLoc, ldc);	
			}
			bp = aml_tiling_tilestart(&tilingB, b, k + 1);
			
		}
		

		ap = aml_tiling_tilestart(&tilingB, a, i+1);
		bp = aml_tiling_tilestart(&tilingB, b, 0);
		cp = aml_tiling_tilestart(&tilingB, c, i+1);
	}
	
}

int argoMM(int argc, char* argv[]){
		AML_BINDING_SINGLE_DECL(binding);
		AML_ARENA_JEMALLOC_DECL(arena);
		AML_DMA_LINUX_SEQ_DECL(dma);
		unsigned long nodemask[AML_NODEMASK_SZ];
		aml_init(&argc, &argv);
		esize = (unsigned long)MEMSIZE/sizeof(double);
		numRows = (unsigned long)sqrt(esize);
	
		#pragma omp parallel
		{
			numthreads = omp_get_num_threads();
			tilesz = pow( numRows / numthreads,2) * sizeof(double); 
			if(argc == 4){
				tilesz = sizeof(double)*(atol(argv[3]) * atol(argv[3]));
			}
			numTiles = MEMSIZE / tilesz; 
			CHUNKING = numTiles / numthreads;
			esz = tilesz/sizeof(double);
		}
	
			
		if(DEBUG || VERBOSE)printf("The total memory size is: %lu\nWe are dealing with a %lu x %lu matrix multiplication\nThe number of threads: %d\nThe chunking is: %lu\nThe tilesz is: %lu\nThat means there are %lu elements per tile\nThere are %lu tiles total\nThe length of a column in bytes is: %lu\n", MEMSIZE, (unsigned long)sqrt(MEMSIZE/sizeof(unsigned long)), (unsigned long)sqrt(MEMSIZE/sizeof(unsigned long)),numthreads, CHUNKING, tilesz, esz, numTiles, rowLengthInBytes);

		assert(!aml_binding_init(&binding, AML_BINDING_TYPE_SINGLE, 0));
		assert(!aml_tiling_init(&tiling, AML_TILING_TYPE_2D, (unsigned long)(sqrt(tilesz/sizeof(double))*sizeof(double)), (unsigned long)(sqrt(tilesz/sizeof(double))*sizeof(double)), tilesz, MEMSIZE));	
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
		AML_NODEMASK_ZERO(nodemask);
		AML_NODEMASK_SET(nodemask, 1);
		assert(!aml_area_linux_init(&fast,
					    AML_AREA_LINUX_MANAGER_TYPE_SINGLE,
					    AML_AREA_LINUX_MBIND_TYPE_REGULAR,
					    AML_AREA_LINUX_MMAP_TYPE_ANONYMOUS,
					    &arena, MPOL_BIND, nodemask));
		
		
		/* allocation */
		//5.33333 GB
		if(MEMSIZE <= 5726623061){
			a = aml_area_malloc(&fast, MEMSIZE);
			b = aml_area_malloc(&fast, MEMSIZE);
			c = aml_area_malloc(&fast, MEMSIZE);
		}
		//8GB, put B & C in Fast
		else if(MEMSIZE <= 8589934592){
			a = aml_area_malloc(&slow, MEMSIZE);
			b = aml_area_malloc(&fast, MEMSIZE);
			c = aml_area_malloc(&fast, MEMSIZE);
		}
		//16GB, put C in Fast
		else if(MEMSIZE <= 17179869184){
			a = aml_area_malloc(&slow, MEMSIZE);
			b = aml_area_malloc(&slow, MEMSIZE);
			c = aml_area_malloc(&fast, MEMSIZE);
		}
		//Put all in slow
		else{
			a = aml_area_malloc(&slow, MEMSIZE);
			b = aml_area_malloc(&slow, MEMSIZE);
			c = aml_area_malloc(&slow, MEMSIZE);
		}
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

		//Prints RDTSC then CLOCK
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
	

		if(MEMSIZE <= 5726623061){
			aml_area_free(&fast, a);
		}
		else{
			aml_area_free(&slow, a);
		}

		if(MEMSIZE > 8589934592){
			aml_area_free(&slow, b);
		}
		else{
			aml_area_free(&fast, b);
		}
		if(MEMSIZE > 17179869184){
			aml_area_free(&slow, c);
		}
		else{
			aml_area_free(&fast, c);
		}
		aml_area_linux_destroy(&slow);
		aml_area_linux_destroy(&fast);
		aml_tiling_destroy(&tiling, AML_TILING_TYPE_1D);
		aml_tiling_destroy(&tilingB, AML_TILING_TYPE_2D);
		aml_binding_destroy(&binding, AML_BINDING_TYPE_SINGLE);
		aml_finalize();
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
		rowLengthInBytes = (unsigned long)sqrt(MEMSIZE);
	}
	else if(argc == 2){	
		if(VERBOSE) printf("Setting number of threads\n");
		omp_set_num_threads(NUMBER_OF_THREADS);
	 	if(VERBOSE) printf("Setting MEMSIZE\n");	
		MEMSIZE = sizeof(double)*(atol(argv[1]) * atol(argv[1]));
		rowLengthInBytes = atol(argv[1]) * sizeof(double);
		if(VERBOSE) printf("1 argument provided, setting numThreads = %d and Memsize = %lu\n", NUMBER_OF_THREADS, MEMSIZE);
	}
	else if(argc >= 3){
		omp_set_num_threads(atoi(argv[2]));
		MEMSIZE = sizeof(double)*(atol(argv[1]) * atol(argv[1]));
		rowLengthInBytes = atol(argv[1]) * sizeof(double);
		if(VERBOSE) printf("Two arguments provided, setting numThreads = %d and Elements per Row = %lu\n", atoi(argv[2]), atol(argv[1]));
	}

	argoMM(argc, argv);

	return 0;
}
