#include <aml.h>
#include <assert.h>
#include <errno.h>
#include <cblas.h>
#include <lapacke.h>
//#include <mkl.h>
//#include <mkl_scalapack.h>
#include <omp.h>
#include <pthread.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>

unsigned long tilesz, tileElements, memsize, tilesPerRow, N;
long long int potrfTime, trsmTime, syrkTime, gemmTime;

int main(int argc, char *argv[])
{
	
	AML_ARENA_JEMALLOC_DECL(arns);
	AML_ARENA_JEMALLOC_DECL(arnf);

	AML_AREA_LINUX_DECL(fast);
	struct bitmask *slowb, *fastb;
	double *array, *b, *c;

	struct timespec start, stop;
	struct timespec start0, stop0;
	struct timespec start1, stop1;
	struct timespec start2, stop2;

	aml_init(&argc, &argv);
	fastb = numa_parse_nodestring_all(argv[1]);
	N = atol(argv[3]);

	int runs;

	if(N < 256){
		runs = 100000;
	}
	else if(N < 1024){
		runs = 1000;	
	}
	else if(N < 4096){
		runs = 10;
	}
	else{
		runs = 5;
	}
	
 	memsize = sizeof(double)*N*N;
	assert(!aml_arena_jemalloc_init(&arns, AML_ARENA_JEMALLOC_TYPE_REGULAR));
	assert(!aml_arena_jemalloc_init(&arnf, AML_ARENA_JEMALLOC_TYPE_REGULAR));

	assert(!aml_area_linux_init(&fast,
				    AML_AREA_LINUX_MANAGER_TYPE_SINGLE,
				    AML_AREA_LINUX_MBIND_TYPE_REGULAR,
				    AML_AREA_LINUX_MMAP_TYPE_ANONYMOUS,
				    &arnf, MPOL_BIND, fastb->maskp));
	array = aml_area_malloc(&fast, memsize);
	b = aml_area_malloc(&fast, memsize);
	c = aml_area_malloc(&fast, memsize);

	
	assert(array != NULL);

	double alpha = 1.0, beta = 1.0;
	for(unsigned long i = 0; i < N*N; i++){
		array[i] = (double)1.0;
		b[i] = (double)1.0;
		c[i] = (double)1.0;
	}

for(int q = 0; q < runs; q++){ 	
	clock_gettime(CLOCK_REALTIME, &start);
	LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', N, array, N); 
	clock_gettime(CLOCK_REALTIME, &stop);
	potrfTime += (stop.tv_nsec - start.tv_nsec) + 1e9*(stop.tv_sec - start.tv_sec);
}

for(int q = 0; q < runs; q++){ 	
	clock_gettime(CLOCK_REALTIME, &start0);
	cblas_dtrsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, N, N, alpha, array, N, b, N);		
	clock_gettime(CLOCK_REALTIME, &stop0);
	trsmTime +=  (stop0.tv_nsec - start0.tv_nsec) + 1e9*(stop0.tv_sec - start0.tv_sec);
}

for(int q = 0; q < runs; q++){ 	
	clock_gettime(CLOCK_REALTIME, &start1);
	cblas_dsyrk(CblasRowMajor, CblasLower, CblasNoTrans, N, N, alpha, array, N, beta, b, N);
	clock_gettime(CLOCK_REALTIME, &stop1);
	syrkTime +=  (stop1.tv_nsec - start1.tv_nsec) + 1e9*(stop1.tv_sec - start1.tv_sec);
}


for(int q = 0; q < runs; q++){ 	
	clock_gettime(CLOCK_REALTIME, &start2);	
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, alpha, array, N, b, N, beta, c, N);		
	clock_gettime(CLOCK_REALTIME, &stop2);
	gemmTime +=  (stop2.tv_nsec - start2.tv_nsec) + 1e9*(stop2.tv_sec - start2.tv_sec);
}

	printf("%lld\t\t\t%lld\t\t\t%lld\t\t\t%lld\n", potrfTime/runs, trsmTime/runs, syrkTime/runs, gemmTime/runs);
	
	aml_area_free(&fast, array);
	aml_area_linux_destroy(&fast);
	aml_finalize();
	return 0;
}
