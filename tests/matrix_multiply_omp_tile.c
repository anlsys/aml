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

#define CBLAS 1 //This will choose whether to use clblas dgemm or our kernel
#define VERBOSE 0 //Verbose mode will print out extra information about what is happening
#define DEBUG 0 //This will print out verbose messages and debugging statements
#define PRINT_ARRAYS 0 
#define HBM 1 
#define DEBUG2 0 
#define INTEL 0
#define ARGONNE 1 
#define BILLION 1000000000L

#if ARGONNE == 1
	#include <aml.h>
	struct aml_tiling tiling, tilingB;
	struct aml_area slow, fast;
	struct aml_scratch sa, sb;

#endif

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
unsigned long long forkingTime, totalForkTime;
struct timespec startClock, endClock;
double elapsedTime;


//AML_TILING_2D_DECL(tiling);
//AML_TILING_2D_DECL(tilingB);
//AML_AREA_LINUX_DECL(slow);
//AML_AREA_LINUX_DECL(fast);
//AML_SCRATCH_PAR_DECL(sa);
//AML_SCRATCH_PAR_DECL(sb);

//This code will take cycles executed as a use for timing the kernel.

unsigned long rdtsc(){
    unsigned int lo,hi;
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return ((unsigned long)hi << 32) | lo;
}


#if ARGONNE == 1
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
	totalWait = 0;
	totalForkTime = 0;
	if(DEBUG2) printf("Inside do_work()\n");	
	int i, k, ai, bi, oldai, oldbi, tilesPerCol;
	MKL_INT lda = (int)rowSizeOfTile, ldb, ldc;
	ldb = lda;
	ldc = lda;
	double *ap, *bp, *cp;
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
	if(DEBUG2) printf("tilesPerCol = %d\n Beginning outerloop\n", tilesPerCol);
	for(i = 0; i < rowSizeInTiles; i++) {
		
		//Request next column of tiles from A into the scratchpad for A	
		if(HBM){
			oldbi = bi;
			oldai = ai;
			aml_scratch_async_pull(&sa, &ar, abaseptr, &ai, a, i + 1);
		 	aml_scratch_async_pull(&sb, &br, bbaseptr, &bi, b, i + 1);
			if(DEBUG) printf("Async pulling next columns of B and A (%d)\n", i);	
		}
		forkingTime = rdtsc();		
		//This will begin the dispersion of work accross threads, this loop is actually O(1) 
		#pragma omp parallel for
		for(k = 0; k < numthreads; k++)
		{
			if(k == numthreads - 1){
				forkingTime = rdtsc() - forkingTime;
				totalForkTime += forkingTime;
				if(DEBUG) printf("forkingTime = %lu\n", forkingTime);
			}
			int j, l, offset;
			double *apLoc, *bpLoc;
			//This loop will cover if threads are handling multiple rows of tiles. This shouldn't be a necessarily large number
			//This loop is technically O(n) but in reality will usually be O(1) because tilesPerCol will be a small number relative to rowSizeInTiles
			for(j = 0; j < tilesPerCol; j++){
				offset = (k * tilesPerCol) + j;
				if(DEBUG2){
					printf("Thread %d has an offset value of %d\n", k, offset);
				}
				//This will give the beginning offset for where each thread should point to in the tilingB sized ap array
				apLoc = aml_tiling_tilestart(&tiling, ap, offset);
				offset = (k * tilesPerCol) + j;
						
				//Now we will iterate through all the tiles in the row of B tiles and compute a partial matrix multiplication
				//This loop is O(n)
				for(l = 0; l < rowSizeInTiles; l++){
					
					bpLoc = aml_tiling_tilestart(&tiling, bp, l);
					
					//This will begin at the tile row that is at x cordinate equal to bp offset, and y cordinate equal to ap offset
					offset = (((int)rowSizeInTiles * ((k * tilesPerCol)+ j) ) + l);
					if(DEBUG2){
						printf("Thread %d is beginning tile c at tile offset<%d> = (rowSizeInTiles<%d> * ((k<%d> * tilesPerCol<%d>) + j<%d>)) + l<%d>\n", k, offset, (int)rowSizeInTiles, k, tilesPerCol, j, l);
					} 
					cp = aml_tiling_tilestart(&tiling, c, offset);
					
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
					//if(DEBUG && k == 0) printf("Beginning matrix multiply\n");
					//Currently we will call the user written kernel, but we will eventually use dgemm???
					//
					#if CBLAS == 1
						if(DEBUG2){
							printf("Beginning cblas dgemm\n");
							fflush(stdout);
						}
						cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ldc, lda, ldb, 1.0, apLoc, lda, bpLoc, ldb, 1.0, cp, ldc);
					#else
						multiplyTiles(apLoc, bpLoc, cp, rowSizeOfTile, rowSizeOfTile);
					#endif
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
			waitingTime = rdtsc();
			aml_scratch_wait(&sa, ar);
			aml_scratch_wait(&sb, br);
			totalWait += rdtsc() - waitingTime;
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

int argoMM(int argc, char* argv[]){
		printf("\n\n-----RUNNING ARGONNE MEMORY OPTIMIZED MATRIX MULTIPLICATION-----\n");
		AML_BINDING_SINGLE_DECL(binding);
		AML_ARENA_JEMALLOC_DECL(arena);
		AML_DMA_LINUX_SEQ_DECL(dma);
		unsigned long nodemask[AML_NODEMASK_SZ];
		aml_init(&argc, &argv);
	
		AML_TILING_2D_DECL(tilingTemp);
		AML_TILING_2D_DECL(tilingBTemp);
		AML_AREA_LINUX_DECL(slowTemp);
		AML_AREA_LINUX_DECL(fastTemp);
		AML_SCRATCH_PAR_DECL(saTemp);
		AML_SCRATCH_PAR_DECL(sbTemp);

		tiling = tilingTemp;
		tilingB = tilingBTemp;
		slow = slowTemp;
		fast = fastTemp;
		sa = saTemp;
		sb = sbTemp;
	
		/* use openmp env to figure out how many threads we want
		 * (we actually use 3x as much)
		 */
		esize = (unsigned long)MEMSIZE/sizeof(double);
		numRows = (unsigned long)sqrt(esize);
	
		if(VERBOSE);
		#pragma omp parallel
		{
			rowLengthInBytes = (unsigned long)sqrt(MEMSIZE/sizeof(double)) * sizeof(double);
			numthreads = omp_get_num_threads();
			//CHUNKING = Total number of columns that will be handled by each thread
			//Tilesz is found by dividing the length (in number of elements) of a dimension by the number of threads.
			//It then checks to see if the tile is too large. If it is, then the size will be reduced until it will fit inside the L2 Cache
			tilesz = ((unsigned long) pow( ( ( sqrt(MEMSIZE / sizeof(double)) ) / (numthreads * 2) ), 2) ) * sizeof(double);
			double multiplier = 2;
			while(tilesz > L2_CACHE_SIZE && argc != 4){
				tilesz = (unsigned long)pow(((unsigned long)sqrt(tilesz/sizeof(double))/2), 2) * sizeof(double); 			
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
		
		assert(!aml_dma_linux_seq_init(&dma, 2));
		//assert(!aml_dma_linux_seq_init(&dma, numthreads*2));
		if(HBM){
			if(DEBUG)printf("Declaring scratchpad for sa\n");
			assert(!aml_scratch_par_init(&sa, &fast, &slow, &dma, &tilingB,
					     2, 2));
			//assert(!aml_scratch_par_init(&sa, &fast, &slow, &dma, &tilingB,
			//		     2*numthreads, numthreads));
	
			if(DEBUG)printf("Declaring scratchpad for sb\n");
			assert(!aml_scratch_par_init(&sb, &fast, &slow, &dma, &tilingB,
					     2, 2));
			//assert(!aml_scratch_par_init(&sb, &fast, &slow, &dma, &tilingB,
					    // 2*numthreads, numthreads));
	
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
		//This will execute on core 0
		clock_gettime(CLOCK_REALTIME, &startClock);
		beginTime = rdtsc();
	
		//BEGIN MULTIPLICATION
		//	#pragma omp parallel for
		//	for(unsigned long i = 0; i < numthreads; i++){
		do_work();
		//	}
		//END MULTIPLICATION
		//This will execute on core 0
		endTime = rdtsc();
		clock_gettime(CLOCK_REALTIME, &endClock);

		elapsedTime = BILLION * ( endClock.tv_sec - startClock.tv_sec ) + (( stop.tv_nsec - start.tv_nsec ));
		printf("Kernel Timing Statistics:\nRDTSC: %lu cycles\nCLOCK: %lf ns\nCycles waiting on memory: %lu\nCycles waiting to fork: %lu\n", endTime - beginTime, elapsedTime, totalWait, totalForkTime);
	
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
	
		if(!correct){
			printf("The matrix multiplication failed. The last incorrect result is at location C(0,0) = %lf in the C matrix\n", c[0]);
		}
		else{
			printf("The matrix multiplication suceeded, C(0,0) = %lf\n", c[0]);
		}
	
	

		if(HBM) aml_scratch_par_destroy(&sa);
		if(HBM) aml_scratch_par_destroy(&sb);
		if(HBM) aml_dma_linux_seq_destroy(&dma);
		aml_area_free(&slow, a);
		aml_area_free(&slow, b);
		if(HBM) aml_area_free(&fast, c);
		if(!HBM) aml_area_free(&slow, c);
		aml_area_linux_destroy(&slow);
		if(HBM) aml_area_linux_destroy(&fast);
		aml_tiling_destroy(&tiling, AML_TILING_TYPE_1D);
		aml_tiling_destroy(&tilingB, AML_TILING_TYPE_2D);
		aml_binding_destroy(&binding, AML_BINDING_TYPE_SINGLE);
		aml_finalize();
}
#endif



#if INTEL == 1
	int intelMM(int argc, char* argv[]){
		printf("\n\n-----RUNNING INTEL MKL MATRIX MULTIPLICATION-----\n");
		double *aIntel, *bIntel, *cIntel;
		unsigned int m,n,p;
		unsigned long intelNumRows, intelEsize;
		intelNumRows = (unsigned long)sqrt(MEMSIZE/8);
		intelEsize = intelNumRows * intelNumRows;
		m = n = p = intelNumRows;

		//printf("Declaring a, b, and c matrices\n");
		aIntel = (double *)mkl_malloc( m*p*sizeof( double ), 64 );
		bIntel = (double *)mkl_malloc( p*n*sizeof( double ), 64 );
		cIntel = (double *)mkl_malloc( m*n*sizeof( double ), 64 );
		//aIntel = (double *)malloc(m*p*sizeof(double));
		//bIntel = (double *)malloc(p*n*sizeof(double));
		//cIntel = (double *)malloc(m*n*sizeof(double));
		if (aIntel == NULL || bIntel == NULL || cIntel == NULL) {
			printf( "\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
			free(aIntel);
			free(bIntel);
			free(cIntel);
			return 1;
		}
		//printf("Initializing values in the matrices\n");
		double alpha = 1.0, beta = 1.0;
		for(unsigned long i = 0; i < intelEsize; i++){
			//printf("%lu ", i);
			aIntel[i] = 1.0;
			bIntel[i] = 1.0;
			cIntel[i] = 0.0;
		}
		//printf("m = %d, n = %d, p = %d, alpha = %lf, beta = %lf, aIntel = %p, bIntel = %p, cIntel = %p\n", m, n , p, alpha, beta, aIntel, bIntel, cIntel);
		printf("Beginning cblas\n");
		mkl_set_num_threads(atoi(argv[2]));

		clock_gettime(CLOCK_REALTIME, &startClock);
		beginTime = rdtsc();
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, p, alpha, aIntel, p, bIntel, n, beta, cIntel, n);
		//This will execute on core 0
		endTime = rdtsc();
		clock_gettime(CLOCK_REALTIME, &endClock);
		elapsedTime = BILLION * ( endClock.tv_sec - startClock.tv_sec ) + (( endClock.tv_nsec - startClock.tv_nsec ) );

		printf("Intel Timing Statistics:\nRDTSC: %lu cycles\nCLOCK: %lf ns\n", endTime - beginTime, elapsedTime);

		return 0; 
	}

#endif

//This matrix multiplication will implement matrix multiplication in the following way:
//	The A, B, and C matrices will be broken into tiles that edge as close to 1 MB as possible (size of L2 cache) while also allowing all threads to work on data.
//	The algorithm will chunk up the work dependent on number of tiles. The multiplication will go as follows:
//		1) The algorithm will take a column of the tiles of A. (Prefetch next column of A tiles to fast memory).
//		2) The algorithm will take a column of B tiles (prefetch next column into fast memory).
//		3) Perform partial matrix multiplications using A tile and B Tile (using dgemm eventually). 
//		4) Repeat 2 & 3 until columns of B tiles are exhausted. Then continue
//		5) Repeat from 1 until columns of A tiles are exhausted. Then continue
//		6) DONE
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
		MEMSIZE = atol(argv[1]);
		if(VERBOSE) printf("1 argument provided, setting numThreads = %d and Memsize = %lu\n", NUMBER_OF_THREADS, MEMSIZE);
	}
	else if(argc >= 3){
		omp_set_num_threads(atoi(argv[2]));
		MEMSIZE = atol(argv[1]);
		if(VERBOSE) printf("Two arguments provided, setting numThreads = %d and Memsize = %lu\n", atoi(argv[2]), atol(argv[1]));

	}



	#if INTEL == 1
		intelMM(argc, argv);	
	#endif

	#if ARGONNE == 1
		argoMM(argc, argv);
	#endif
	


	return 0;
}
