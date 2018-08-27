#include <aml.h>
#include <assert.h>
#include <errno.h>
#include <openblas.h>
//#include <mkl.h>
//#include <mkl_scalapack.h>
#include <omp.h>
#include <pthread.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>

unsigned long tilesz, tileElements, memsize, tilesPerRow, N, T;

AML_TILING_2D_CONTIG_ROWMAJOR_DECL(tiling_row);
AML_AREA_LINUX_DECL(slow);
AML_AREA_LINUX_DECL(fast);

int cholOMP(double* L){
	#pragma omp parallel
	#pragma omp master
	for(int k = 0; k < tilesPerRow; k++){
		#pragma omp task depend(inout:L[(k*tilesPerRow*tileElements + k*tileElements) : tileElements])
		{
			//async pull ext k,k tile (do I depend on L or the scratchpad?)
			 LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', T, &L[(k*tilesPerRow + k*tileElements)], T); 
			//async push the result (maybe not???)
		
		}
		
		for(int m = k + 1; m < tilesPerRow; m++)
			#pragma omp task depend(in:L[(k*tilesPerRow*tileElements + k*tileElements) : tileElements]) depend(inout:L[(m*tilesPerRow*tileElements + k*tileElements) : tileElements])
			{ 	
				//prefetch tile m,k and use tile k,k which should be in HBM already	
				cblas_dtrsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, T, T, 1.0, &L[(k*tilesPerRow*tileElements + k*tileElements)], T, &L[m*tilesPerRow*tileElements + k*tileElements], T);
				//push the result (maybe not since it gets used later?)
			}
		for(int m = k + 1; m < tilesPerRow; m++){
			#pragma omp task depend(in:L[(m*tilesPerRow*tileElements + k*tileElements) : tileElements]) depend(inout:L[(m*tilesPerRow*tileElements + m*tileElements) : tileElements])
			{
				//Async pull next m,m iteration and m,k should be in already
				cblas_dsyrk(CblasRowMajor, CblasLower, CblasNoTrans, T, T, 1.0, &L[m*tilesPerRow*tileElements + k*tileElements], T, 1.0, &L[m*tilesPerRow*tileElements + m*tileElements], T);
				//push the results, (maybe not since other iterations need it)
			}
			for(int i = k + 1; i < m; i++)
				#pragma omp task depend(in:L[(m*tilesPerRow*tileElements + i*tileElements) : tileElements]) \
					depend(inout:L[(m*tilesPerRow*tileElements + i*tileElements) : tileElements])
				{ 
					//Prefetch m,k i,k and m,i
					cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, T, T, T, 1.0, &L[m*tilesPerRow*tileElements + k*tileElements], T, &L[i*tilesPerRow*tileElements + k*tileElements], T, 1.0, &L[m*tilesPerRow*tileElements + i*tileElements], T);
					//Not sure what to do

				}
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
	double *a;
	aml_init(&argc, &argv);
	fastb = numa_parse_nodestring_all(argv[1]);
	slowb = numa_parse_nodestring_all(argv[2]);
	N = atol(argv[3]);
	T = atol(argv[4]);
 	memsize = sizeof(double)*N*N;
	tilesz = sizeof(double)*T*T;
	tileElements = T * T;
	tilesPerRow = N / T;
	
	//Ensures no funny threading is happening within openblas kernels.
	openblas_set_num_threads(1);


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

	assert(!aml_tiling_init(&tiling_row, AML_TILING_TYPE_2D_CONTIG_ROWMAJOR,
				tilesz, memsize, N/T , N/T));

	assert(a != NULL);

	double alpha = 1.0, beta = 1.0;
	for(unsigned long i = 0; i < N*N; i++){
		a[i] = (double)1.0;
	}

	clock_gettime(CLOCK_REALTIME, &start);

	int index = 0;
	char uplo = 'L';
	int descA[9];

	clock_gettime(CLOCK_REALTIME, &stop);
	long long int time = 0;
	double flops;

	clock_gettime(CLOCK_REALTIME, &start0);
		
	cholOMP(a);		

	clock_gettime(CLOCK_REALTIME, &stop0);
	time =  (stop0.tv_nsec - start0.tv_nsec) +
                1e9* (stop0.tv_sec - start0.tv_sec);
	flops = ((pow(N, 3.0)/3)/(time/1e9));
	printf("cholesky-aml: %llu %lld %lld %lf\n", N, memsize, time, flops/1e9);	

	aml_area_free(&fast, a);
	aml_area_linux_destroy(&slow);
	aml_area_linux_destroy(&fast);
	aml_finalize();
	return 0;
}
