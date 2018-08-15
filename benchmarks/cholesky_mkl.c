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


int cholOMP(double* L, unsigned long n){
	unsigned long i, j, k;
        omp_lock_t writelock;
        omp_init_lock(&writelock);
        
        for (j = 0; j < n; j++) {
                
		for (i = 0; i < j; i++){                        
                        L[i*n + j] = 0;
                }
                                
                #pragma omp parallel for shared(L) private(k)
                for (k = 0; k < i; k++) {
                        omp_set_lock(&writelock);
                        L[j*n + j] = L[j*n + j] - L[j*n + k] * L[j*n + k]; //Critical section.
                        omp_unset_lock(&writelock);
                }                        
                
                #pragma omp single        
                L[i*n + i] = sqrt(L[j*n + j]);        
                
                #pragma omp parallel for shared(L) private(i, k)
                for (i = j+1; i < n; i++) {
                        for (k = 0; k < j; k++) {
                                L[i*n + j] = L[i*n + j] - L[i*n + k] * L[j*n +k];
                        }
                        L[i*n + j] = L[i*n + j] / L[j*n + j];
                }                
        
        }

	omp_destroy_lock(&writelock);

	return 0;
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
