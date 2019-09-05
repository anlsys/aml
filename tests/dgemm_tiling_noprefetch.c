#include <aml.h>
#include <aml/area/linux.h>
#include <aml/layout/dense.h>
#include <aml/layout/reshape.h>
#include <aml/layout/native.h>
#include <aml/layout/pad.h>
#include <aml/tiling/resize.h>
#include <assert.h>

void larp_8_62_8_31_477(const int32_t nblockm, const int32_t nblockn, const int32_t nblockk, const double *a, const double*b, double *c);


#define USED_TH		60
#define NUM_INNER_TH	2
#define NUM_OUTER_TH	(USED_TH/NUM_INNER_TH)

#define MR 31
#define NR 8
#define KB 477
#define NBLOCKA 2
#define NBLOCKM NUM_OUTER_TH
#define NBLOCKN 294
#define NBLOCKK 4
#define M (MR*NBLOCKA*NBLOCKM)
#define N (NR*NBLOCKN)
#define K (KB*NBLOCKK)

int main(int argc, char *argv[]) {
	aml_init(&argc, &argv);
	int block_number;
        int block_count;

	struct aml_bitmap slowb, fastb;
	struct aml_area *slow, *fast;
	struct aml_layout *a_l, *b_l, *c_l;
	struct aml_layout *a_l_fast, *b_l_fast, *c_l_fast;
	struct aml_tiling *a_t, *b_t, *c_t;

	assert(aml_bitmap_from_string(&fastb, "1") == 0);
	assert(aml_bitmap_from_string(&slowb, "0") == 0);
        block_number = atol(argv[1]);
        block_count = block_number*block_number;
        

	aml_area_linux_create(&slow, &slowb, AML_AREA_LINUX_POLICY_BIND);
	assert(slow != NULL);
	aml_area_linux_create(&fast, &fastb, AML_AREA_LINUX_POLICY_BIND);
	assert(fast != NULL);

	double *a = (double *)aml_area_mmap(slow, block_count*M*K, NULL);
	double *b = (double *)aml_area_mmap(slow, block_count*K*N, NULL);
        double *c = (double *)aml_area_mmap(slow, block_count*M*N, NULL);
	double *a_fast = (double *)aml_area_mmap(fast, M*K, NULL);
	double *b_fast = (double *)aml_area_mmap(fast, K*N, NULL);
        double *c_fast = (double *)aml_area_mmap(fast, M*N, NULL);

        size_t a_dims[] = {block_number*M, block_number*K};
        size_t b_dims[] = {block_number*K, block_number*N};
        size_t c_dims[] = {block_number*M, block_number*N};

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

        

	for(int i = 0; i < block_number; i++) {
		for(int j = 0; j < block_number; j++) {
			struct aml_layout *c_tile;
			struct aml_layout *c_reshape_tile;
			c_tile = aml_tiling_index(c_t, (size_t[]){i, j});
			aml_layout_reshape(c_tile, &c_reshape_tile, 5, c_reshape_tile_dims);
			aml_copy_layout_transform_native(c_l_fast, c_reshape_tile, &c_transpose[0]);
			for(int k = 0; k < block_number; k++) {
				struct aml_layout *a_tile, *b_tile;
				struct aml_layout *a_reshape_tile, *b_reshape_tile;
				a_tile = aml_tiling_index(a_t, (size_t[]){i, k});
				b_tile = aml_tiling_index(b_t, (size_t[]){k, j});
				aml_layout_reshape(a_tile, &a_reshape_tile,
						   5, a_reshape_tile_dims);
				aml_layout_reshape(b_tile, &b_reshape_tile,
						   4, b_reshape_tile_dims);
				aml_copy_layout_transform_native(a_l_fast, a_reshape_tile, a_transpose);
				aml_copy_layout_transform_native(b_l_fast, b_reshape_tile, b_transpose);
				larp_8_62_8_31_477(NBLOCKM, NBLOCKN, NBLOCKK, a_fast, b_fast, c_fast);
			}
			aml_copy_layout_transform_native(c_reshape_tile, c_l_fast, c_transpose_back);
		}
	}
	return 0;
}
