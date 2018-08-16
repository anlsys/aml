#include <aml.h>
#include <assert.h>
#include <errno.h>
#include <mkl.h>
#include <omp.h>
#include <pthread.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>

unsigned long tilesize;


int cholOMP(double* L, unsigned long n){
	#pragma omp parallel
	for(int k = 0; k < n; k++){
		#pragma omp task depend(inout:L[(k*n + k):n])
		{ LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', n, &L[k*n + k], n); }
		for(int m = k + 1; m < n; m++)
			#pragma omp task depend(in:L[(k*n + k):n]) depend(inout:L[(m*n + k):n])
			{ cblas_dtrsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, n, n, 1.0, &L[k*n + k], n, &L[m*n + k, n], n);}
		for(int m = k + 1; m < n; m++){
			#pragma omp task depend(in:L[(m*n + k):n]) depend(inout:L[(m*n + m):n])
			{cblas_dsyrk(CblasRowMajor, CblasLower, CblasNoTrans, n, n, 1.0, &L[m*n + k], n, 1.0, &L[m*n + m], n);}
			for(int i = k + 1; i < m; i++)
				#pragma omp task depend(in:L[(m*n + i):n]) \
					depend(inout:L[(m*n + i):n])
				{ cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, &L[m*n + k], n, &L[i * n + k], n, 1.0, &L[m*n + i], n);}
		}
	}
}


int main(int argc, char *argv[])
{
	AML_ARENA_JEMALLOC_DECL(arns);
	AML_ARENA_JEMALLOC_DECL(arnf);
	AML_AREA_LINUX_DECL(slow);
	AML_AREA_LINUX_DECL(fast);
	struct bitmask *slowb, *fastb;
	struct timespec start, stop;
	struct timespec start0, stop0;
	double *a, *b, *c;
	aml_init(&argc, &argv);
	fastb = numa_parse_nodestring_all(argv[1]);
	slowb = numa_parse_nodestring_all(argv[2]);
	unsigned long N = atol(argv[3]);
	unsigned long memsize = sizeof(double)*N*N;

	assert(!aml_arena_jemalloc_init(&arns, AML_ARENA_JEMALLOC_TYPE_REGULAR));
	assert(!aml_area_linux_init(&slow,
				    AML_AREA_LINUX_MANAGER_TYPE_SINGLE,
				    AML_AREA_LINUX_MBIND_TYPE_REGULAR,
				    AML_AREA_LINUX_MMAP_TYPE_ANONYMOUS,
				    &arns, MPOL_BIND, slowb->maskp));
	assert(!aml_arena_jemalloc_init(&arnf, AML_ARENA_JEMALLOC_TYPE_REGULAR));
	assert(!aml_area_linux_init(&fast,
				    AML_AREA_LINUX_MANAGER_TYPE_SINGLE,
				    AML_AREA_LINUX_MBIND_TYPE_REGULAR,
				    AML_AREA_LINUX_MMAP_TYPE_ANONYMOUS,
				    &arnf, MPOL_BIND, fastb->maskp));
	a = aml_area_malloc(&fast, memsize);
	b = aml_area_malloc(&fast, memsize);
	assert(a != NULL && b != NULL && c != NULL);

	double alpha = 1.0, beta = 1.0;
	for(unsigned long i = 0; i < N*N; i++){
		a[i] = (double)1.0;
		b[i] = (double)1.0;
	}

	clock_gettime(CLOCK_REALTIME, &start);

	LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', N, a, N);

	clock_gettime(CLOCK_REALTIME, &stop);
	long long int time = 0;
	time =  (stop.tv_nsec - start.tv_nsec) +
                1e9* (stop.tv_sec - start.tv_sec);
	double flops = ((pow(N,3.0)/3)/(time/1e9));
	/* print the flops in GFLOPS */
	printf("cholesky-mkl: %llu %lld %lld %lf\n", N, memsize, time, flops/1e9);


	clock_gettime(CLOCK_REALTIME, &start0);
		
	cholOMP(b, N);		

	clock_gettime(CLOCK_REALTIME, &stop0);
	time =  (stop0.tv_nsec - start0.tv_nsec) +
                1e9* (stop0.tv_sec - start0.tv_sec);
	flops = ((pow(N, 3.0)/3)/(time/1e9));
	printf("cholesky-aml: %llu %lld %lld %lf\n", N, memsize, time, flops/1e9);	

	//aml_area_free(&fast, a);
	//aml_area_free(&slow, b);
	//aml_area_linux_destroy(&slow);
	//aml_area_linux_destroy(&fast);
	//aml_finalize();
	return 0;
}
