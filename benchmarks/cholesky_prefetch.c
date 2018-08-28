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

unsigned long tilesz, tileElements, memsize, tilesPerRow, N, T;
AML_DMA_LINUX_SEQ_DECL(dma);
AML_TILING_2D_CONTIG_ROWMAJOR_DECL(tiling_row);
AML_AREA_LINUX_DECL(slow);
AML_AREA_LINUX_DECL(fast);
AML_SCRATCH_SEQ_DECL(sa);
double *a, *b;
struct bitmask *slowb, *fastb;
struct timespec start, stop;
struct timespec start0, stop0;
int **tracker;	

int cholOMP(){
	#pragma omp parallel
	#pragma omp master
	for(int k = 0; k < tilesPerRow; k++){

		#pragma omp task depend(in:a[(k*tilesPerRow*tileElements + k*tileElements) : tileElements]) depend (out:b[(k*tilesPerRow*tileElements) : tileElements])
		{
			if(tracker[k][k] != 1){
				int q = k*tilesPerRow + k - 1;
				aml_scratch_pull(&sa, b, &q, a, (k*tilesPerRow) + k);
				tracker[k][k] = 1;
			}	
		}		

		#pragma omp task depend(inout:b[(k*tilesPerRow*tileElements + k*tileElements) : tileElements])
		{
			 LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', T, &b[(k*tilesPerRow + k*tileElements)], T); 
		}
		
		for(int m = k + 1; m < tilesPerRow; m++){
			#pragma omp task depend(in:a[(m*tilesPerRow*tileElements + k*tileElements) : tileElements], a[(k*tilesPerRow*tileElements + k*tileElements) : tileElements]) depend(out:b[(m*tilesPerRow*tileElements + k*tileElements) : tileElements], b[(k*tilesPerRow*tileElements + k*tileElements) : tileElements])
			{
				if(tracker[m][k] != 1){
					int q = m*tilesPerRow + k - 1;
					aml_scratch_pull(&sa, b, &q, a, (m*tilesPerRow) + k);
					tracker[m][k] = 1;
				}
				if(tracker[k][k] != 1){
					int q = k*tilesPerRow + k - 1;
					aml_scratch_pull(&sa, b, &q, a, (k*tilesPerRow) + k);
					tracker[k][k] = 1;
				}		
			}

			#pragma omp task depend(in:b[(k*tilesPerRow*tileElements + k*tileElements) : tileElements]) depend(inout:b[(m*tilesPerRow*tileElements + k*tileElements) : tileElements])
			{ 		
				cblas_dtrsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, T, T, 1.0, &b[(k*tilesPerRow*tileElements + k*tileElements)], T, &b[m*tilesPerRow*tileElements + k*tileElements], T);
			}
		}
			

		for(int m = k + 1; m < tilesPerRow; m++){

			#pragma omp task depend(in:a[(m*tilesPerRow*tileElements + k*tileElements) : tileElements], a[(m*tilesPerRow*tileElements + m*tileElements) : tileElements]) depend(out:b[(m*tilesPerRow*tileElements + k*tileElements) : tileElements], b[(m*tilesPerRow*tileElements + m*tileElements) : tileElements])
			{
				if(tracker[m][k] != 1){
					int q = m*tilesPerRow + k - 1;
					aml_scratch_pull(&sa, b, &q, a, (m*tilesPerRow) + k);
					tracker[m][k] = 1;
				}
				if(tracker[m][m] != 1){
					int q = m*tilesPerRow + m - 1;
					aml_scratch_pull(&sa, b, &q, a, (m*tilesPerRow) + m);
					tracker[m][m] = 1;
				}
			} 

			#pragma omp task depend(in:b[(m*tilesPerRow*tileElements + k*tileElements) : tileElements]) depend(inout:b[(m*tilesPerRow*tileElements + m*tileElements) : tileElements])
			{
				cblas_dsyrk(CblasRowMajor, CblasLower, CblasNoTrans, T, T, 1.0, &b[m*tilesPerRow*tileElements + k*tileElements], T, 1.0, &b[m*tilesPerRow*tileElements + m*tileElements], T);
			}
			for(int i = k + 1; i < m; i++){
			
				#pragma omp task depend(in:a[(m*tilesPerRow*tileElements + k*tileElements) : tileElements], a[(i*tilesPerRow*tileElements + k*tileElements) : tileElements], a[(m*tilesPerRow*tileElements + i*tileElements) : tileElements]) depend(out:b[(m*tilesPerRow*tileElements + k*tileElements) : tileElements], b[(i*tilesPerRow*tileElements + k*tileElements) : tileElements], b[(m*tilesPerRow*tileElements + i*tileElements) : tileElements])
				{
				
					if(tracker[m][k] != 1){
						int q = m*tilesPerRow + k - 1;
						aml_scratch_pull(&sa, b, &q, a, (m*tilesPerRow) + k);

						tracker[m][k] = 1;
					}
					if(tracker[i][k] != 1){
						int q = i*tilesPerRow + k - 1;
						aml_scratch_pull(&sa, b, &q, a, (i*tilesPerRow) + k);
						tracker[i][k] = 1;
					}
					if(tracker[m][i] != 1){
						int q = m*tilesPerRow + i - 1;
						aml_scratch_pull(&sa, b, &q, a, (m*tilesPerRow) + i);
						tracker[m][i] = 1;
					}

				}				

				#pragma omp task depend(in:b[(m*tilesPerRow*tileElements + k*tileElements) : tileElements], b[(i*tilesPerRow*tileElements + k*tileElements) : tileElements]) depend(inout:b[(m*tilesPerRow*tileElements + i*tileElements) : tileElements])
				{ 
					cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, T, T, T, 1.0, &b[m*tilesPerRow*tileElements + k*tileElements], T, &b[i*tilesPerRow*tileElements + k*tileElements], T, 1.0, &b[m*tilesPerRow*tileElements + i*tileElements], T);
				}
			}
		}
	}
}


int main(int argc, char *argv[])
{
	AML_ARENA_JEMALLOC_DECL(arns);
	AML_ARENA_JEMALLOC_DECL(arnf);
	
	aml_init(&argc, &argv);
	fastb = numa_parse_nodestring_all(argv[1]);
	slowb = numa_parse_nodestring_all(argv[2]);
	N = atol(argv[3]);
	T = atol(argv[4]);
 	memsize = sizeof(double)*N*N;
	tilesz = sizeof(double)*T*T;
	tileElements = T * T;
	tilesPerRow = N / T;
	
	//Allocate a tracker for determining if it is in HBM
	tracker = (int**)malloc(tilesPerRow*sizeof(int*));
	for(int i = 0; i < tilesPerRow; i++){
		tracker[i] = (int*)calloc(tilesPerRow, sizeof(int));
	}

	
	//Ensures no funny threading is happening within openblas kernels.
	//openblas_set_num_threads(1);

	assert(!aml_tiling_init(&tiling_row, AML_TILING_TYPE_2D_CONTIG_ROWMAJOR, tilesz, memsize, N/T , N/T));


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
	assert(!aml_dma_linux_seq_init(&dma, 64));

	assert(!aml_scratch_seq_init(&sa, &fast, &slow, &dma, &tiling_row, (size_t)(tilesPerRow*tilesPerRow), (size_t)64));

	
	a = aml_area_malloc(&slow, memsize);
	b = aml_scratch_baseptr(&sa);

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
		
	cholOMP();		

	clock_gettime(CLOCK_REALTIME, &stop0);
	time =  (stop0.tv_nsec - start0.tv_nsec) +
                1e9* (stop0.tv_sec - start0.tv_sec);
	flops = ((pow(N, 3.0)/3)/(time/1e9));
	printf("cholesky-aml: %llu %lld %lld %lf\n", N, memsize, time, flops/1e9);	

	aml_dma_linux_seq_destroy(&dma);
	aml_area_free(&fast, a);
	aml_area_free(&slow, b);
	aml_scratch_seq_destroy(&sa);
	aml_area_linux_destroy(&slow);
	aml_area_linux_destroy(&fast);
	aml_tiling_destroy(&tiling_row, AML_TILING_TYPE_2D_CONTIG_ROWMAJOR);
	aml_finalize();
	return 0;
}
