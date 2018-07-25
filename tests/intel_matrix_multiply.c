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


unsigned long MEMSIZE;
struct timespec startClock, endClock;
double elapsedTime;
unsigned long long beginTime, endTime;


//This code will take cycles executed as a use for timing the kernel.

unsigned long rdtsc(){
    unsigned int lo,hi;
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return ((unsigned long)hi << 32) | lo;
}



int intelMM(int argc, char* argv[]){
	double *aIntel, *bIntel, *cIntel;
	unsigned int m,n,p;
	unsigned long intelNumRows, intelEsize;
	intelNumRows = (unsigned long)sqrt(MEMSIZE/8);
	intelEsize = intelNumRows * intelNumRows;
	m = n = p = intelNumRows;
	aIntel = (double *)mkl_malloc( m*p*sizeof( double ), 64 );
	bIntel = (double *)mkl_malloc( p*n*sizeof( double ), 64 );
	cIntel = (double *)mkl_malloc( m*n*sizeof( double ), 64 );
	if (aIntel == NULL || bIntel == NULL || cIntel == NULL) {
		printf( "\n ERROR: Can't allocate memory for matrices. Aborting... \n");
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

	clock_gettime(CLOCK_REALTIME, &startClock);
	beginTime = rdtsc();
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, p, alpha, aIntel, p, bIntel, n, beta, cIntel, n);
	//This will execute on core 0
	endTime = rdtsc();
	clock_gettime(CLOCK_REALTIME, &endClock);
	elapsedTime = BILLION * ( endClock.tv_sec - startClock.tv_sec ) + (( endClock.tv_nsec - startClock.tv_nsec ) );
	//This will print the RDTSC measurement followed by CLOCK
	printf("%lu\t%lf\n", endTime - beginTime, elapsedTime);
	return 0; 
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
		mkl_set_num_threads(NUMBER_OF_THREADS);
		MEMSIZE = MEMSIZES;
	}
	else if(argc == 2){	
		if(VERBOSE) printf("Setting number of threads\n");
		mkl_set_num_threads(NUMBER_OF_THREADS);
	 	if(VERBOSE) printf("Setting MEMSIZE\n");	
		MEMSIZE = sizeof(double)*(atol(argv[1]) * atol(argv[1]));
		if(VERBOSE) printf("1 argument provided, setting numThreads = %d and Memsize = %lu\n", NUMBER_OF_THREADS, MEMSIZE);
	}
	else if(argc >= 3){
		mkl_set_num_threads(atoi(argv[2]));
		MEMSIZE = sizeof(double)*(atol(argv[1]) * atol(argv[1]));
		if(VERBOSE) printf("Two arguments provided, setting numThreads = %d and Memsize = %lu\n", atoi(argv[2]), atol(argv[1]));
	}

	intelMM(argc, argv);

	return 0;
}
