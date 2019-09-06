#include <aml.h>
#include <aml/area/linux.h>
#include <aml/layout/dense.h>
#include <aml/layout/reshape.h>
#include <aml/layout/native.h>
#include <aml/layout/pad.h>
#include <aml/tiling/resize.h>
#include <assert.h>
#include <stdlib.h>
#include <cblas.h>

void larp_8_62_8_31_477(const int32_t nblockm, const int32_t nblockn, const int32_t nblockk, const double *a, const double*b, double *c);


#define USED_TH		60
#define NUM_INNER_TH	2
#define NUM_OUTER_TH	(USED_TH/NUM_INNER_TH)

#define MR 31
#define NR 8
#define KB 477
#define NBLOCKA 2
#define NBLOCKM NUM_OUTER_TH*2
#define NBLOCKN 294*2
#define NBLOCKK 4*2
#define M (MR*NBLOCKA*NBLOCKM)
#define N (NR*NBLOCKN)
#define K (KB*NBLOCKK)


void init_matrix(double *ptr, size_t size) {
	for(size_t i = 0; i < size; i++) {
		ptr[i] = ((double)random()*2/(double)RAND_MAX)-1.0;
	}
}

#define PITCH_A_0_S	KB
#define PITCH_A_1_S	KB*NBLOCKK
#define PITCH_A_2_S	KB*NBLOCKK*MR
#define PITCH_A_3_S	KB*NBLOCKK*MR*NBLOCKA

#define PITCH_A_0_D	MR
#define PITCH_A_1_D	MR*KB
#define PITCH_A_2_D	MR*KB*NBLOCKA
#define PITCH_A_3_D	MR*KB*NBLOCKA*NBLOCKM

void transform_a(size_t block_number, double * restrict dst, const double * restrict src) {
	for(int i = 0; i < NBLOCKM; i++)
	for(int j = 0; j < NBLOCKA; j++)
	for(int k = 0; k < MR; k++)
	for(int l = 0; l < NBLOCKK; l++)
	for(int m = 0; m < KB; m++)
		dst[k + m*PITCH_A_0_D + j*PITCH_A_1_D + i*PITCH_A_2_D + l*PITCH_A_3_D] =
		src[m + l*PITCH_A_0_S + (k*PITCH_A_1_S + j*PITCH_A_2_S + i*PITCH_A_3_S)*block_number];
}

#define PITCH_B_0_S	NR
#define PITCH_B_1_S	NR*NBLOCKN
#define PITCH_B_2_S	NR*NBLOCKN*KB

#define PITCH_B_0_D	NR
#define PITCH_B_1_D	NR*KB
#define PITCH_B_2_D	NR*KB*NBLOCKN
void transform_b(size_t block_number, double * restrict dst, const double * restrict src) {
	for( int i = 0; i < NBLOCKK; i++)
	for( int j = 0; j < KB; j++)
	for( int k = 0; k < NBLOCKN; k++)
	for( int l = 0; l < NR; l++)
		dst[l + j*PITCH_B_0_D + k*PITCH_B_1_D + i*PITCH_B_2_D] =
		src[l + k*PITCH_B_0_S + (j*PITCH_B_1_S + i*PITCH_B_2_S)*block_number];
}

#define PITCH_C_0_S	NR
#define PITCH_C_1_S     NR*NBLOCKN
#define PITCH_C_2_S     NR*NBLOCKN*MR
#define PITCH_C_3_S     NR*NBLOCKN*MR*NBLOCKA

#define PITCH_C_0_D     NR
#define PITCH_C_1_D     NR*MR
#define PITCH_C_2_D     NR*MR*NBLOCKA
#define PITCH_C_3_D     NR*MR*NBLOCKA*NBLOCKN

void transform_c(size_t block_number, double * restrict dst, const double * restrict src) {
	for(int i = 0; i < NBLOCKM; i++)
	for(int j = 0; j < NBLOCKA; j++)
	for(int k = 0; k < MR; k++)
	for(int l = 0; l < NBLOCKN; l++)
	for(int m = 0; m < NR; m++)
		dst[m + k*PITCH_C_0_D + j*PITCH_C_1_D + l*PITCH_C_2_D + i*PITCH_C_3_D] =
		src[m + l*PITCH_C_0_S + (k*PITCH_C_1_S + j*PITCH_C_2_S + i*PITCH_C_3_S)*block_number];
}

void transform_c_reverse(size_t block_number, double * restrict dst, const double * restrict src) {
	for(int i = 0; i < NBLOCKM; i++)
	for(int j = 0; j < NBLOCKN; j++)
	for(int k = 0; k < NBLOCKA; k++)
	for(int l = 0; l < MR; l++)
	for(int m = 0; m < NR; m++)
		dst[m + j*PITCH_C_0_S + (l*PITCH_C_1_S + k*PITCH_C_2_S + i*PITCH_C_3_S)*block_number] =
		src[m + l*PITCH_C_0_D + k*PITCH_C_1_D + j*PITCH_C_2_D + i*PITCH_C_3_D];
}

int main(int argc, char *argv[]) {
	aml_init(&argc, &argv);
	int block_number;
        int block_count;
	int do_test = 0;

	struct aml_bitmap slowb, fastb;
	struct aml_area *slow, *fast;
	struct aml_layout *a_l, *b_l, *c_l;
	struct aml_layout *a_l_fast, *b_l_fast, *c_l_fast;
	struct aml_tiling *a_t, *b_t, *c_t;

	assert(aml_bitmap_from_string(&fastb, "1") == 0);
	assert(aml_bitmap_from_string(&slowb, "0") == 0);
        if (argc > 1) {
        	block_number = atol(argv[1]);
	} else {
		block_number = 2;
		do_test= 1;
	}
        block_count = block_number*block_number;
        

	aml_area_linux_create(&slow, &slowb, AML_AREA_LINUX_POLICY_BIND);
	assert(slow != NULL);
	aml_area_linux_create(&fast, &fastb, AML_AREA_LINUX_POLICY_BIND);
	assert(fast != NULL);

	size_t big_m = block_number * M;
	size_t big_n = block_number * N;
	size_t big_k = block_number * K;

	double *a = (double *)aml_area_mmap(slow, sizeof(double)*big_m*big_k, NULL);
	double *b = (double *)aml_area_mmap(slow, sizeof(double)*big_k*big_n, NULL);
        double *c = (double *)aml_area_mmap(slow, sizeof(double)*big_m*big_n, NULL);
        double *c_ref = (double *)aml_area_mmap(slow, sizeof(double)*big_m*big_n, NULL);
	double *a_fast = (double *)aml_area_mmap(fast, sizeof(double)*M*K, NULL);
	double *b_fast = (double *)aml_area_mmap(fast, sizeof(double)*K*N, NULL);
        double *c_fast = (double *)aml_area_mmap(fast, sizeof(double)*M*N, NULL);

        init_matrix(a, big_m*big_k);
	init_matrix(b, big_k*big_n);
	init_matrix(c, big_m*big_n);
	larp_8_62_8_31_477(NBLOCKM, NBLOCKN, NBLOCKK, a, b, c_ref);
        memcpy(c_ref, c, sizeof(double)*big_m*big_n);

        size_t a_dims[] = {big_m, big_k};
        size_t b_dims[] = {big_k, big_n};
        size_t c_dims[] = {big_m, big_n};

        size_t a_tile_dims[] 		= {M, K};
	size_t a_reshape_tile_dims[] 	= {NBLOCKM, NBLOCKA, MR, NBLOCKK, KB};
	size_t a_transpose_tile_dims[] 	= {NBLOCKK, NBLOCKM, NBLOCKA, KB, MR};
//	size_t a_transpose[]		= {3, 0, 1, 4, 2};
	size_t a_transpose[]		= {2, 0, 3, 4, 1};
        size_t b_tile_dims[] 		= {K, N};
        size_t b_reshape_tile_dims[] 	= {NBLOCKK, KB, NBLOCKN, NR};
        size_t b_transpose_tile_dims[] 	= {NBLOCKK, NBLOCKN, KB, NR};
//	size_t b_transpose[]		= {0, 2, 1, 3};
	size_t b_transpose[]		= {0, 2, 1, 3};
        size_t c_tile_dims[] 		= {M, N};
        size_t c_reshape_tile_dims[] 	= {NBLOCKM, NBLOCKA, MR, NBLOCKN, NR};
        size_t c_transpose_tile_dims[] 	= {NBLOCKM, NBLOCKN, NBLOCKA, MR, NR};
//	size_t c_transpose[]	 	= {0, 3, 1, 2, 4};
//	size_t c_transpose_back[] 	= {0, 2, 3, 1, 4};
        size_t c_transpose[]	 	= {0, 2, 3, 1, 4};
        size_t c_transpose_back[] 	= {0, 3, 1, 2, 4};

        aml_layout_dense_create(&a_l, a, AML_LAYOUT_ORDER_ROW_MAJOR,
				sizeof(double),
				2, a_dims, NULL, NULL);
        aml_layout_dense_create(&b_l, b, AML_LAYOUT_ORDER_ROW_MAJOR,
				sizeof(double),
				2, b_dims, NULL, NULL);
        aml_layout_dense_create(&c_l, c, AML_LAYOUT_ORDER_ROW_MAJOR,
				sizeof(double),
				2, c_dims, NULL, NULL);

        aml_layout_dense_create(&a_l_fast, a_fast, AML_LAYOUT_ORDER_ROW_MAJOR,
				sizeof(double),
				5, a_transpose_tile_dims, NULL, NULL);
        aml_layout_dense_create(&b_l_fast, b_fast, AML_LAYOUT_ORDER_ROW_MAJOR,
				sizeof(double),
				4, b_transpose_tile_dims, NULL, NULL);
        aml_layout_dense_create(&c_l_fast, c_fast, AML_LAYOUT_ORDER_ROW_MAJOR,
				sizeof(double),
				5, c_transpose_tile_dims, NULL, NULL);

        assert( !aml_tiling_resize_create(&a_t, AML_LAYOUT_ORDER_ROW_MAJOR, a_l, 2, a_tile_dims) );
        assert( !aml_tiling_resize_create(&b_t, AML_LAYOUT_ORDER_ROW_MAJOR, b_l, 2, b_tile_dims) );
        assert( !aml_tiling_resize_create(&c_t, AML_LAYOUT_ORDER_ROW_MAJOR, c_l, 2, c_tile_dims) );

        
	struct timespec start, stop;
	clock_gettime(CLOCK_REALTIME, &start);
	for (int i = 0; i < block_number; i++) {
		for (int j = 0; j < block_number; j++) {
			struct aml_layout *c_tile;
			struct aml_layout *c_reshape_tile;
			c_tile = aml_tiling_index(c_t, (size_t[]){i, j});
			aml_layout_reshape(c_tile, &c_reshape_tile, 5, c_reshape_tile_dims);
			//aml_copy_layout_transform_native(c_l_fast, c_reshape_tile, c_transpose);
			transform_c(block_number, c_fast, aml_layout_deref(c_reshape_tile,(size_t[]){0,0,0,0,0}));
			for (int k = 0; k < block_number; k++) {
				struct aml_layout *a_tile, *b_tile;
				struct aml_layout *a_reshape_tile, *b_reshape_tile;
				a_tile = aml_tiling_index(a_t, (size_t[]){i, k});
				b_tile = aml_tiling_index(b_t, (size_t[]){k, j});
				aml_layout_reshape(a_tile, &a_reshape_tile,
						   5, a_reshape_tile_dims);
				aml_layout_reshape(b_tile, &b_reshape_tile,
						   4, b_reshape_tile_dims);
				//aml_copy_layout_transform_native(a_l_fast, a_reshape_tile, a_transpose);
				transform_a(block_number, a_fast, aml_layout_deref(a_reshape_tile,(size_t[]){0,0,0,0,0}));
				//aml_copy_layout_transform_native(b_l_fast, b_reshape_tile, b_transpose);
				transform_b(block_number, b_fast, aml_layout_deref(b_reshape_tile,(size_t[]){0,0,0,0}));
				larp_8_62_8_31_477(NBLOCKM, NBLOCKN, NBLOCKK, a_fast, b_fast, c_fast);
			}
			//aml_copy_layout_transform_native(c_reshape_tile, c_l_fast, c_transpose_back);
			transform_c_reverse(block_number, aml_layout_deref(c_reshape_tile,(size_t[]){0,0,0,0,0}), c_fast);
		}
	}
	clock_gettime(CLOCK_REALTIME, &stop);
	double time = 0;
	time =  (stop.tv_nsec - start.tv_nsec) +
                1e9* (stop.tv_sec - start.tv_sec);
	double flops = (2.0*big_n*big_m*big_k)/(time/1e9);

	printf("dgemm        : %lf ms %f GFlops/s\n", time/1e6, flops/1e9);

	clock_gettime(CLOCK_REALTIME, &start);
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, big_m, big_n, big_k, 1.0, a, big_k, b, big_n, 1.0, c_ref, big_n);
	clock_gettime(CLOCK_REALTIME, &stop);
        cblas_daxpy(big_n*big_m, -1.0, c, 1, c_ref, 1);
        double max_error;
	size_t max_error_index;
        max_error_index = cblas_idamax(big_n*big_m, c_ref, 1);
        max_error = c_ref[max_error_index];
	time =  (stop.tv_nsec - start.tv_nsec) +
                1e9* (stop.tv_sec - start.tv_sec);
	flops = (2.0*big_n*big_m*big_k)/(time/1e9);
	/* print the flops in GFLOPS */
	printf("dgemm-vanilla: %lf ms %f GFlops/s %le max error\n", time/1e6, flops/1e9, max_error);

	return 0;
}
