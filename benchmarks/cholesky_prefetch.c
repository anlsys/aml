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
AML_TILING_2D_ROWMAJOR_DECL(tiling_row);
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

		if(tracker[k][k] == -1){
				//printf("Pulling tracker[%d][%d] for potrf\n", k, k);
				aml_scratch_pull(&sa, b, &tracker[k][k], a, aml_tiling_tileid(&tiling_row, k, k));	
		}		

		#pragma omp task depend(in:tracker[k][k])
		{
			//printf("Asserting tracker for potrf\n");
			assert(tracker[k][k] != -1);
			double* bTile = aml_tiling_tilestart(&tiling_row, b, tracker[k][k]);
			LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', T, bTile, T); 
		}
		
		for(int m = k + 1; m < tilesPerRow; m++){

			if(tracker[m][k] == -1){
				//printf("Pulling tracker[%d][%d] for trsm\n", m, k);
				aml_scratch_pull(&sa, b, &tracker[m][k], a, aml_tiling_tileid(&tiling_row, m, k));
			}

			#pragma omp task depend(in:tracker[m][k], tracker[k][k])
			{ 

				//printf("Asserting trackers for dtrsm\n");
				assert(tracker[m][k] != -1);
				assert(tracker[k][k] != -1);
				double* bTile0 = aml_tiling_tilestart(&tiling_row, b, tracker[m][k]);
				double* bTile1 = aml_tiling_tilestart(&tiling_row, b, tracker[k][k]);
				cblas_dtrsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, T, T, 1.0, bTile0, T, bTile1, T);
			}
		}
			

		for(int m = k + 1; m < tilesPerRow; m++){

			if(tracker[m][m] == -1){
					//printf("Pulling tracker[%d][%d] for syrk\n", m, m);
					aml_scratch_pull(&sa, b, &tracker[m][m], a, aml_tiling_tileid(&tiling_row, m, m));
			}

		
			#pragma omp task depend(in:tracker[m][k], tracker[m][m])
			{

				//printf("Asserting trackers for syrk\n");
				assert(tracker[m][k] != -1);
				assert(tracker[m][m] != -1);
				double* bTile0 = aml_tiling_tilestart(&tiling_row, b, tracker[m][k]);
				double* bTile1 = aml_tiling_tilestart(&tiling_row, b, tracker[m][m]);
				cblas_dsyrk(CblasRowMajor, CblasLower, CblasNoTrans, T, T, 1.0, bTile0, T, 1.0, bTile1, T);
			}
			for(int i = k + 1; i < m; i++){
				
				if(tracker[m][k] == -1){
					//printf("Pulling tracker[%d][%d] for gemm\n", m, k);
					aml_scratch_pull(&sa, b, &tracker[m][k], a, aml_tiling_tileid(&tiling_row, m, k));
				}
				if(tracker[i][k] == -1){
					//printf("Pulling tracker[%d][%d] for gemm\n", i, k);
					aml_scratch_pull(&sa, b, &tracker[i][k], a, aml_tiling_tileid(&tiling_row, i, k));
				}
				if(tracker[m][i] == -1){
					//printf("Pulling tracker[%d][%d] for gemm\n", m, i);
					aml_scratch_pull(&sa, b, &tracker[m][i], a, aml_tiling_tileid(&tiling_row, m, i));
				}
				

				#pragma omp task depend(in:tracker[m][k], tracker[i][k], tracker[m][i])
				{

					//printf("Asserting trackers for gemm\n");
					assert(tracker[m][k] != -1);
					assert(tracker[i][k] != -1);
					assert(tracker[m][i] != -1);
					double* bTile0 = aml_tiling_tilestart(&tiling_row, b, tracker[m][k]);
					double* bTile1 = aml_tiling_tilestart(&tiling_row, b, tracker[i][k]);
 					double* bTile2 = aml_tiling_tilestart(&tiling_row, b, tracker[m][i]);
					cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, T, T, T, 1.0, bTile0, T, bTile1, T, 1.0, bTile2, T);
				}
			}
		}



	//#pragma omp task depend(inout:tracker[0:k][0:k])
	{
	//	printf("Done with loop iteration k = %d\n", k);
	//	//send back the row and col
	//	for(int i = 0; i<k; i++){
	//		//Send a row tile
	//		aml_scratch_push(&sa, b, &tracker[i][k], a, (i*tilesPerRow) + k);
	//		tracker[i][k] = -1;
	//		//Send a col tile
	//		aml_scratch_push(&sa, b, &tracker[k][i], a, (k*tilesPerRow) + i);
	//		tracker[k][i] = -1;
	//	}
	//	//send back the diagonal
	//	aml_scratch_push(&sa, b, &tracker[k][k], a, (k*tilesPerRow) + k);
	//	tracker[k][k] = -1;
	}
		
	
	
	}
}


int main(int argc, char *argv[])
{
	AML_ARENA_JEMALLOC_DECL(arns);
	AML_ARENA_JEMALLOC_DECL(arnf);
	AML_DMA_LINUX_SEQ_DECL(dma);

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
	for(int i = 0; i < tilesPerRow; i++){
		for(int j = 0; j <tilesPerRow; j++){
			tracker[i][j] = -1;
		}
	}

	
	//Ensures no funny threading is happening within openblas kernels.
	//openblas_set_num_threads(1);

	assert(!aml_tiling_init(&tiling_row, AML_TILING_TYPE_2D_ROWMAJOR, tilesz, memsize, N/T , N/T));


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
	assert(!aml_dma_linux_seq_init(&dma, 4));

	printf("Attempting to make scratchpad in fast mem\n");
	assert(!aml_scratch_seq_init(&sa, &fast, &slow, &dma, &tiling_row, (size_t)(tilesPerRow*tilesPerRow), 2));
	printf("Sucessfully created a scratchpad in fast mem\n");
	
	a = aml_area_malloc(&slow, memsize);
	b = aml_scratch_baseptr(&sa);

	assert(a != NULL && b != NULL);

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
	aml_area_free(&slow, a);
	aml_scratch_seq_destroy(&sa);
	aml_area_linux_destroy(&slow);
	aml_area_linux_destroy(&fast);
	aml_tiling_destroy(&tiling_row, AML_TILING_TYPE_2D_ROWMAJOR);
	aml_finalize();
	return 0;
}
