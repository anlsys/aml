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
#define DEBUG 1 //This will print out verbose messages and debugging statements
#define PRINT_ARRAYS 0 
#define BILLION 1000000000L

#include <aml.h>

AML_TILING_2D_DECL(tiling);
AML_TILING_2D_DECL(tilingB);
AML_AREA_LINUX_DECL(slow);
AML_AREA_LINUX_DECL(fast);
AML_SCRATCH_PAR_DECL(sa);
AML_SCRATCH_PAR_DECL(sb);
AML_SCRATCH_PAR_DECL(sc);

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
	int i, k, ai, bi, ci, oldai, oldbi, oldci, pushCi, pullCi, tilesPerCol;
	int lda = (int)rowSizeOfTile, ldb, ldc;
	ldb = lda;
	ldc = lda;
	double *ap, *bp, *cp;
	void *abaseptr, *bbaseptr, *cbaseptr;
	int colSizeInTiles = rowSizeInTiles;
       
	abaseptr = aml_scratch_baseptr(&sa);
	bbaseptr = aml_scratch_baseptr(&sb);
	cbaseptr = aml_scratch_baseptr(&sc);
	
	ai = -1; bi = -1, ci = -1;
	
	ap = a;
	bp = b;
	cp = c;

	struct aml_scratch_request *ar, *br, *crPull, *crPush;
		
	
	//This is performing 7) in the comments, it will end when A and C tiles are all done		
	for(i = 0; i < colSizeInTiles; i++) {
		//Request next column of tiles from A into the scratchpad for A	
		int l, k;
		oldai = ai;
		aml_scratch_async_pull(&sa, &ar, abaseptr, &ai, a, i + 1);
		aml_scratch_async_pull(&sc, &crPull, cbaseptr, &ci, c, i + 1);
		//This loop will go through each row of B and each tile in a row of A
		for(k = 0; k < rowSizeInTiles; k++){
			double *apLoc;
			oldbi = bi;
			if(k = rowSizeInTiles - 1){
				aml_scratch_async_pull(&sb, &br, &bi, b, 0);
			}
			else{
				aml_scratch_async_pull(&sb, &br, &bi, b, k+1);
			}

			apLoc = aml_tiling_tilestart(&tiling, ap, k);
			#pragma omp parallel for
			for(l = 0; l < rowSizeInTiles; l++){
				double *bpLoc, *cpLoc;
				cpLoc = aml_tiling_tilestart(&tiling, cp, l);			
				bpLoc = aml_tiling_tilestart(&tiling, bp, l);
				cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ldc, lda, ldb, 1.0, apLoc, lda, bpLoc, ldb, 1.0, cpLoc, ldc);	
			}
			aml_scratch_wait(&sb, br);
			bp = aml_tiling_tilestart(&tilingB, bbaseptr, bi);
			aml_scratch_release(&sb, oldbi);
			
			
		}
		
		if(i != 0){
			aml_scratch_wait(&sc, crPush);
		}
		
		aml_scratch_wait(&sc, crPull);		
		aml_scratch_wait(&sa, ar);
		
		oldci = pushCi;
		pushCi = ci;
		ci = pullCi;
		pullCi = oldci;

		aml_scratch_async_push(&sc, &crPush, cbaseptr, &pushCi, c, i);
		ap = aml_tiling_tilestart(&tilingB, abaseptr, ai);
		cp = aml_tiling_tilestart(&tilingB, cbaseptr, ci);

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
			rowLengthInBytes = (unsigned long)sqrt(MEMSIZE/sizeof(double)) * sizeof(double);
			numthreads = omp_get_num_threads();
			tilesz = ((unsigned long) pow( ( ( (unsigned long)sqrt(MEMSIZE / sizeof(double)) ) / numthreads ), 2) ) * sizeof(double);
			double multiplier = 2;
			if(argc == 4){
				tilesz = sizeof(double)*(atol(argv[3]) * atol(argv[3]));
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

		AML_NODEMASK_ZERO(nodemask);
		AML_NODEMASK_SET(nodemask, 1);
		assert(!aml_area_linux_init(&fast,
					    AML_AREA_LINUX_MANAGER_TYPE_SINGLE,
					    AML_AREA_LINUX_MBIND_TYPE_REGULAR,
					    AML_AREA_LINUX_MMAP_TYPE_ANONYMOUS,
					    &arena, MPOL_BIND, nodemask));
		
		assert(!aml_dma_linux_seq_init(&dma, 4));

		if(DEBUG) printf("Init A\n");
		//assert(!aml_scratch_par_init(&sa, &fast, &slow, &dma, &tilingB, 2, 1));
		if(DEBUG) printf("Init B\n");
		assert(!aml_scratch_par_init(&sb, &fast, &slow, &dma, &tilingB, 2, 1));
		if(DEBUG) printf("Init C\n");
		assert(!aml_scratch_par_init(&sc, &fast, &slow, &dma, &tilingB, 3, 2));
		
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
	
		aml_scratch_par_destroy(&sa);
		aml_scratch_par_destroy(&sb);
		aml_scratch_par_destroy(&sc);
		aml_dma_linux_par_destroy(&dma);
		aml_area_free(&slow, a);
		aml_area_free(&slow, b);
		aml_area_free(&slow, c);
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
