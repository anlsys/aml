#define _GNU_SOURCE
#include <aml.h>
#include <aml/area/linux.h>
#include <aml/layout/dense.h>
#include <aml/layout/reshape.h>
#include <aml/layout/native.h>
#include <aml/layout/pad.h>
#include <aml/tiling/resize.h>
#include <aml/dma/linux-spin.h>
#include <assert.h>
#include <stdlib.h>
#include <cblas.h>
#include <sched.h>

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

int dma_transform_a(struct aml_layout *dest, const struct aml_layout *src, void *arg)
{
	struct aml_layout_dense *d;
	struct aml_layout_dense *s;
	double * restrict dd;
	double * restrict ss;
	size_t * nb;
	nb = (size_t *)arg;
	d = (struct aml_layout_dense *)dest->data;
	dd = (double * restrict)d->ptr;
	s = (struct aml_layout_dense *)src->data;
	ss = (double * restrict)s->ptr;
	transform_a(*nb, dd, ss);
	return 0;
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

int dma_transform_b(struct aml_layout *dest, const struct aml_layout *src, void *arg)
{
	struct aml_layout_dense *d;
	struct aml_layout_dense *s;
	double * restrict dd;
	double * restrict ss;
	size_t * nb;
	nb = (size_t *)arg;
	d = (struct aml_layout_dense *)dest->data;
	dd = (double * restrict)d->ptr;
	s = (struct aml_layout_dense *)src->data;
	ss = (double * restrict)s->ptr;
	transform_b(*nb, dd, ss);
	return 0;
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

int dma_transform_c(struct aml_layout *dest, const struct aml_layout *src, void *arg)
{
	struct aml_layout_dense *d;
	struct aml_layout_dense *s;
	double * restrict dd;
	double * restrict ss;
	size_t * nb;
	nb = (size_t *)arg;
	d = (struct aml_layout_dense *)dest->data;
	dd = (double * restrict)d->ptr;
	s = (struct aml_layout_dense *)src->data;
	ss = (double * restrict)s->ptr;
	transform_c(*nb, dd, ss);
	return 0;
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

int dma_transform_c_reverse(struct aml_layout *dest, const struct aml_layout *src, void *arg)
{
	struct aml_layout_dense *d;
	struct aml_layout_dense *s;
	double * restrict dd;
	double * restrict ss;
	size_t * nb;
	nb = (size_t *)arg;
	d = (struct aml_layout_dense *)dest->data;
	dd = (double * restrict)d->ptr;
	s = (struct aml_layout_dense *)src->data;
	ss = (double * restrict)s->ptr;
	transform_c_reverse(*nb, dd, ss);
	return 0;
}

void larp_layout(struct aml_layout *a, struct aml_layout *b, struct aml_layout *c)
{
	struct aml_layout_dense *al;
	struct aml_layout_dense *bl;
	struct aml_layout_dense *cl;
	double * restrict ad;
	double * restrict bd;
	double * restrict cd;
	al = (struct aml_layout_dense *)a->data;
	ad = (double * restrict)al->ptr;
	bl = (struct aml_layout_dense *)b->data;
	bd = (double * restrict)bl->ptr;
	cl = (struct aml_layout_dense *)c->data;
	cd = (double * restrict)cl->ptr;
	larp_8_62_8_31_477(NBLOCKM, NBLOCKN, NBLOCKK, ad, bd, cd);
}

int nextai(int i, int j, int k, int nB, int *ri, int *rk)
{
	if(k < nB -1)
	{
		*ri = i;
		*rk = k+1;
	}
	else
	{
		if(j < nB -1) {
			*ri = i;
			*rk = 0;
		}
		else
			if(i < nB -1) {
				*ri = i+1;
				*rk = 0;
			} else 
				return -1;
	}
	return 0;
}

int nextbi(int i, int j, int k, int nB, int *rk, int *rj)
{
	if(k < nB -1)
	{
		*rk = k+1;
		*rj = j;
	}
	else
	{
		if(j < nB -1) {
			*rk = 0;
			*rj = j+1;
		} else
			if(i < nB -1) {
				*rk = 0;
				*rj = 0;
			} else
				return -1;
	}
	return 0;
}

int nextci(int i, int j, int nB, int *ri, int *rj)
{
	if(j < nB -1)
	{
		*ri = i;
		*rj = j+1;
	}
	else
	{
		if(i < nB -1)
		{
			*ri = i+1;
			*rj = 0;
		}
		else
			return -1;
	}
	return 0;
}

int flipflop(int i, int b)
{
	return (i+1)%b;
}

#define launch_dma(dma, request, layout, tiling, i, j, ndims, shape, transform) {\
			struct aml_layout *tile; \
			struct aml_layout *reshape_tile; \
			tile = aml_tiling_index(tiling, (size_t[]){i, j}); \
			aml_layout_reshape(tile, &reshape_tile, ndims, shape); \
			aml_dma_async_copy_custom(dma, &request, layout, reshape_tile, transform, &block_number);}

#define launch_reverse_dma(dma, request, layout, tiling, i, j, ndims, shape, transform) {\
			struct aml_layout *tile; \
			struct aml_layout *reshape_tile; \
			tile = aml_tiling_index(tiling, (size_t[]){i, j}); \
			aml_layout_reshape(tile, &reshape_tile, ndims, shape); \
			aml_dma_async_copy_custom(dma, &request, reshape_tile, layout, transform, &block_number);}

int main(int argc, char *argv[]) {
	aml_init(&argc, &argv);
	size_t block_number;
        int block_count;
	int do_test = 0;

	struct aml_bitmap slowb, fastb;
	struct aml_area *slow, *fast;
	struct aml_layout *a_l, *b_l, *c_l;
	struct aml_layout *a_l_fast[2], *b_l_fast[2], *c_l_fast[3];
	struct aml_tiling *a_t, *b_t, *c_t;
	struct aml_dma *dma_a, *dma_b, *dma_cr, *dma_cw;
	struct aml_dma_request *ar, *br, *crr, *crw;

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
	cpu_set_t cpuset;
	CPU_ZERO(&cpuset);
	CPU_SET(60, &cpuset);
	assert( !aml_dma_linux_spin_create(&dma_a, &cpuset, NULL, NULL));
	CPU_ZERO(&cpuset);
	CPU_SET(61, &cpuset);
	assert( !aml_dma_linux_spin_create(&dma_b, &cpuset, NULL, NULL));
	CPU_ZERO(&cpuset);
	CPU_SET(62, &cpuset);
	assert( !aml_dma_linux_spin_create(&dma_cr, &cpuset, NULL, NULL));
	CPU_ZERO(&cpuset);
	CPU_SET(63, &cpuset);
	assert( !aml_dma_linux_spin_create(&dma_cw, &cpuset, NULL, NULL));

	size_t big_m = block_number * M;
	size_t big_n = block_number * N;
	size_t big_k = block_number * K;

	double *a = (double *)aml_area_mmap(slow, sizeof(double)*big_m*big_k, NULL);
	double *b = (double *)aml_area_mmap(slow, sizeof(double)*big_k*big_n, NULL);
        double *c = (double *)aml_area_mmap(slow, sizeof(double)*big_m*big_n, NULL);
        double *c_ref = (double *)aml_area_mmap(slow, sizeof(double)*big_m*big_n, NULL);
	double *a_fast = (double *)aml_area_mmap(fast, 2*sizeof(double)*M*K, NULL);
	double *b_fast = (double *)aml_area_mmap(fast, 2*sizeof(double)*K*N, NULL);
        double *c_fast = (double *)aml_area_mmap(fast, 3*sizeof(double)*M*N, NULL);

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

	for (int i = 0; i < 2; i++) {
        	aml_layout_dense_create(&a_l_fast[i], a_fast + (i*M*K), AML_LAYOUT_ORDER_ROW_MAJOR,
				sizeof(double),
				5, a_transpose_tile_dims, NULL, NULL);
		aml_layout_dense_create(&b_l_fast[i], b_fast + (i*K*N), AML_LAYOUT_ORDER_ROW_MAJOR,
				sizeof(double),
				4, b_transpose_tile_dims, NULL, NULL);
	}
	for (int i = 0; i < 3; i++) {
		aml_layout_dense_create(&c_l_fast[i], c_fast + (i*M*N), AML_LAYOUT_ORDER_ROW_MAJOR,
				sizeof(double),
				5, c_transpose_tile_dims, NULL, NULL);
	}

        assert( !aml_tiling_resize_create(&a_t, AML_LAYOUT_ORDER_ROW_MAJOR, a_l, 2, a_tile_dims) );
        assert( !aml_tiling_resize_create(&b_t, AML_LAYOUT_ORDER_ROW_MAJOR, b_l, 2, b_tile_dims) );
        assert( !aml_tiling_resize_create(&c_t, AML_LAYOUT_ORDER_ROW_MAJOR, c_l, 2, c_tile_dims) );

	struct timespec start, stop;
	clock_gettime(CLOCK_REALTIME, &start);
	int ri = 0, rj = 0, rk = 0;
	int aindex = -1, bindex = -1, cindex = -1;
	int oldaindex = -1, oldbindex = -1, oldcindex = -1;
	aindex = flipflop(aindex, 2);
	bindex = flipflop(bindex, 2);
	cindex = flipflop(cindex, 3);
	launch_dma(dma_a, ar, a_l_fast[aindex], a_t, 0, 0, 5, a_reshape_tile_dims, dma_transform_a); 
	launch_dma(dma_b, br, b_l_fast[bindex], b_t, 0, 0, 4, b_reshape_tile_dims, dma_transform_b); 
	launch_dma(dma_cr, crr, c_l_fast[cindex], c_t, 0, 0, 5, c_reshape_tile_dims, dma_transform_c); 
	for (int i = 0; i < block_number; i++) {
		for (int j = 0; j < block_number; j++) {
			aml_dma_wait(dma_cr, &crr);
			oldcindex = cindex;
			cindex = flipflop(cindex, 3);
			if(nextci(i, j, block_number, &ri, &rj) != -1) {
				launch_dma(dma_cr, crr, c_l_fast[cindex], c_t, ri, rj, 5, c_reshape_tile_dims, dma_transform_c); 
			}	
			for (int k = 0; k < block_number; k++) {
				aml_dma_wait(dma_a, &ar);
				aml_dma_wait(dma_b, &br);
				oldaindex = aindex;
				aindex = flipflop(aindex, 2);
				if(nextai(i, j, k, block_number, &ri, &rk) != -1) {
					launch_dma(dma_a, ar, a_l_fast[aindex], a_t, ri, rk, 5, a_reshape_tile_dims, dma_transform_a);
				}
				oldbindex = bindex;
				bindex = flipflop(bindex, 2);
				if(nextbi(i, j, k, block_number, &rk, &rj) != -1) {
					launch_dma(dma_b, br, b_l_fast[bindex], b_t, rk, rj, 4, b_reshape_tile_dims, dma_transform_b);
				}
				larp_layout(a_l_fast[oldaindex], b_l_fast[oldbindex], c_l_fast[oldcindex]);
			}
			if (i != 0 && j != 0) aml_dma_wait(dma_cw, &crw);
			launch_reverse_dma(dma_cw, crw, c_l_fast[oldcindex], c_t, i, j, 5, c_reshape_tile_dims, dma_transform_c_reverse); 
		}
	}
	aml_dma_wait(dma_cw, &crw);
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
