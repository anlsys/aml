#include <assert.h>
#include <errno.h>
#include <mkl.h>
#include <omp.h>
#include <pthread.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>


int main(int argc, char *argv[])
{
	/* to match the other codes, we expect 3 arguments and use
 	 * the 3rd one for matrix size
 	 */ 
	assert(argc == 4);
	struct timespec start, stop;
	long int N = atol(argv[3]);
	unsigned long memsize = sizeof(double)*N*N;

	double *a, *b, *c;
	a = (double *)mkl_malloc(memsize, 64 );
	b = (double *)mkl_malloc(memsize, 64 );
	c = (double *)mkl_malloc(memsize, 64 );
	assert(a != NULL && b != NULL && c != NULL);

	double alpha = 1.0, beta = 1.0;
	for(unsigned long i = 0; i < N*N; i++){
		a[i] = (double)rand();
		b[i] = (double)rand();
		c[i] = 0.0;
	}

	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, alpha, a, N, b, N, beta, c, N);
	clock_gettime(CLOCK_REALTIME, &start);
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, alpha, a, N, b, N, beta, c, N);
	clock_gettime(CLOCK_REALTIME, &stop);
	long long int time = 0;
	time =  (stop.tv_nsec - start.tv_nsec) +
                1e9* (stop.tv_sec - start.tv_sec);
	double flops = (2.0*N*N*N)/(time/1e9);
	/* print the flops in GFLOPS */
	printf("dgemm-vanilla: %llu %lld %lld %f\n", N, memsize, time, flops/1e9);
	return 0;
}
