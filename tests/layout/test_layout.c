#include <aml.h>
#include <aml/layout/dense.h>
#include <aml/layout/reshape.h>
#include <aml/layout/native.h>
#include <assert.h>

void test_slice_contiguous(void)
{
	int memory[6*5*4];

	size_t dims_col[3]     = {4, 5, 6};
	size_t offsets_col[3]  = {2, 2, 3};
	size_t new_dims_col[3] = {2, 3, 3};

	size_t dims_row[3]     = {6, 5, 4};	
	size_t offsets_row[3]  = {3, 2, 2};
	size_t new_dims_row[3] = {3, 3, 2};

	size_t coords[3];
	void *ptr;
	int val;
	
        int l = 0;
	for(size_t i = 0; i < dims_col[2]; i++)
	for(size_t j = 0; j < dims_col[1]; j++)
	for(size_t k = 0; k < dims_col[0]; k++, l++)
		memory[i*dims_col[1]*dims_col[0] + j*dims_col[0] + k] = l;

	struct aml_layout *a, *b;
	
	assert(aml_layout_dense_create(&a,
				       (void *) memory,
				       AML_LAYOUT_ORDER_COLUMN_MAJOR,
				       sizeof(int),
				       sizeof(dims_col)/sizeof(*dims_col),
				       dims_col,
				       NULL,
				       dims_col) == AML_SUCCESS);

	assert(aml_layout_slice(a, &b, new_dims_col, offsets_col, NULL) == AML_SUCCESS);	
	assert(AML_LAYOUT_ORDER_COLUMN_MAJOR == aml_layout_order(b));

	
	for(size_t i = 0; i < new_dims_col[0]; i++)
	for(size_t j = 0; j < new_dims_col[1]; j++)
	for(size_t k = 0; k < new_dims_col[2]; k++)
	{
		coords[0] = i; coords[1] = j; coords[2] = k;
		
		val = memory[ (i + offsets_col[0]) + 
			      (j + offsets_col[1]) * dims_col[0] +
			      (k + offsets_col[2]) * dims_col[0] * dims_col[1] ];
		
		ptr = aml_layout_deref_safe(b, coords);
		//fprintf(stderr, "%d == %d\n", val, *(int *)ptr);
		assert( val == *(int *)ptr);
	}
	free(a);
	free(b);

	assert(aml_layout_dense_create(&a,
				       (void *) memory,
				       AML_LAYOUT_ORDER_ROW_MAJOR,
				       sizeof(int),
				       sizeof(dims_row)/sizeof(*dims_row),
				       dims_row,
				       NULL,
				       dims_row) == AML_SUCCESS);
	
	assert(aml_layout_slice(a, &b, new_dims_row, offsets_row, NULL) == AML_SUCCESS);
	assert(AML_LAYOUT_ORDER_ROW_MAJOR == aml_layout_order(b));

	for(size_t i = 0; i < new_dims_row[0]; i++)
	for(size_t j = 0; j < new_dims_row[1]; j++)
	for(size_t k = 0; k < new_dims_row[2]; k++)
	{
		coords[0] = i; coords[1] = j; coords[2] = k;
		ptr = aml_layout_deref_safe(b, coords);		
		
		val = memory[ (i + offsets_row[0]) * dims_row[1] * dims_row[2] +
			      (j + offsets_row[1]) * dims_row[2] +
			      (k + offsets_row[2]) ];

		//fprintf(stderr, "%d == %d\n", val, *(int *)ptr);
		assert( val == *(int *)ptr);
		
	}
	free(a);
	free(b);
}

void test_slice_strided(void)
{
	int memory[12][5][8];

	size_t stride[3]         = {2, 1, 2};
	
	size_t dims_col[3]       = {4, 5, 6};
	size_t offsets_col[3]    = {1, 2, 0};
	size_t new_dims_col[3]   = {2, 3, 3};
	size_t new_stride_col[3] = {2, 1, 1};	
	size_t pitch_col[3]      = {8, 5, 12};
	
	size_t dims_row[3]       = {6, 5, 4};	
	size_t pitch_row[3]      = {12, 5, 8};
	size_t offsets_row[3]    = {0, 2, 1};
	size_t new_dims_row[3]   = {3, 3, 2};
	size_t new_stride_row[3] = {1, 1, 2};

	size_t coords[3];
	void *ptr;

        int l = 0;
	for(size_t i = 0; i < 12; i++)
	for(size_t j = 0; j < 5; j++)
	for(size_t k = 0; k < 8; k++, l++)
		memory[i][j][k] = l;

	struct aml_layout *a, *b;

	assert(aml_layout_dense_create(&a,
				       (void *) memory,
				       AML_LAYOUT_ORDER_COLUMN_MAJOR,
				       sizeof(int),
				       3,
				       dims_col,
				       stride,
				       pitch_col) == AML_SUCCESS);

	assert(aml_layout_slice(a, &b, new_dims_col, offsets_col, new_stride_col) == AML_SUCCESS);
	
	for(size_t i = 0; i < 3; i++)
	for(size_t j = 0; j < 3; j++)
	for(size_t k = 0; k < 2; k++){
		coords[0] = k; coords[1] = j; coords[2] = i;
		ptr = aml_layout_deref_safe(b, coords);

		assert( memory[stride[2] * (offsets_col[2] + new_stride_col[2] * i)][
			       stride[1] * (offsets_col[1] + new_stride_col[1] * j)][
			       stride[0] * (offsets_col[0] + new_stride_col[0] * k)] == *(int *)ptr);
	}
	
	free(a);
	free(b);

	assert(aml_layout_dense_create(&a,
				       (void *) memory,
				       AML_LAYOUT_ORDER_ROW_MAJOR,
				       sizeof(int),
				       3,
				       dims_row,
				       stride,
				       pitch_row) == AML_SUCCESS);
	
	assert(aml_layout_slice(a, &b, new_dims_row, offsets_row, new_stride_row) == AML_SUCCESS);

	for(size_t i = 0; i < 3; i++)
	for(size_t j = 0; j < 3; j++)
	for(size_t k = 0; k < 2; k++){
		coords[0] = i; coords[1] = j; coords[2] = k;
		ptr = aml_layout_deref_safe(b, coords);
		
		assert( memory[stride[2] * (offsets_col[2] + new_stride_col[2] * i)][stride[1] * (offsets_col[1] + new_stride_col[1] * j)][stride[0] * (offsets_col[0] + new_stride_col[0] * k)] == *(int *)ptr);
	}

	free(a);
	free(b);

}

void test_reshape_contiguous(void)
{
	int memory[4*5*6];

	size_t dims_col[3] = {4, 5, 6};
	size_t dims_row[3] = {6, 5, 4};

	size_t stride[3] = {1, 1, 1};

	size_t new_dims_col[2] = {24, 5};
	size_t new_dims_row[2] = {5, 24};

	size_t coords[2];
	void *b_ptr, *c_ptr;

	int i;
        for(i = 0; i < 4*5*6; i++)
		memory[i] = i;

	struct aml_layout *a, *b, *c;

	assert(aml_layout_dense_create(&a,
				       (void *) memory,
				       AML_LAYOUT_ORDER_COLUMN_MAJOR,
				       sizeof(int),
				       3,
				       dims_col,
				       stride,
				       dims_col) == AML_SUCCESS);

	assert(aml_layout_reshape(a, &b, 2, new_dims_col) == AML_SUCCESS);
	assert(AML_LAYOUT_ORDER_COLUMN_MAJOR == aml_layout_order(b));
	
	aml_layout_reshape_create(&c,
				  a,
				  AML_LAYOUT_ORDER_COLUMN_MAJOR,
				  2,
				  new_dims_col);
	assert(AML_LAYOUT_ORDER_COLUMN_MAJOR == aml_layout_order(c));
	
	i = 0;
	for(size_t j = 0; j < 5; j++)
	for(size_t k = 0; k < 24; k++, i++) {
		coords[0] = k; coords[1] = j;
		b_ptr = aml_layout_deref_safe(b, coords);
		c_ptr = aml_layout_deref_safe(c, coords);
		assert(i == *(int *)b_ptr);
		assert(i == *(int *)c_ptr);
	}

	free(a);
	free(b);
	free(c);

	assert(aml_layout_dense_create(&a,
				       (void *) memory,
				       AML_LAYOUT_ORDER_ROW_MAJOR,
				       sizeof(int),
				       3,
				       dims_row,
				       stride,
				       dims_row) == AML_SUCCESS);
	assert(aml_layout_reshape(a, &b, 2, new_dims_row) == AML_SUCCESS);
	assert(AML_LAYOUT_ORDER_ROW_MAJOR == aml_layout_order(b));

	aml_layout_reshape_create(&c,
				  a,
				  AML_LAYOUT_ORDER_ROW_MAJOR,
				  2,
				  new_dims_row);
	assert(AML_LAYOUT_ORDER_ROW_MAJOR == aml_layout_order(c));
	
	i = 0;
	for(size_t j = 0; j < 5; j++)
	for(size_t k = 0; k < 24; k++, i++) {
		coords[0] = j; coords[1] = k;
		b_ptr = aml_layout_deref_safe(b, coords);
		c_ptr = aml_layout_deref_safe(c, coords);			
		assert(i == *(int *)b_ptr);
		assert(i == *(int *)c_ptr);
	}

	free(a);
	free(b);
	free(c);
}

void test_reshape_discontiguous(void)
{
	int memory[7][6][5];

	size_t dims_col[3] = {4, 5, 6};
	size_t dims_row[3] = {6, 5, 4};

	size_t stride[3] = {1, 1, 1};

	size_t pitch_col[3] = {5, 6, 7};
	size_t pitch_row[3] = {7, 6, 5};

	size_t new_dims_col[5] = {2, 2, 5, 2, 3};
	size_t new_dims_row[5] = {3, 2, 5, 2, 2};

	size_t coords[5];
	void * ptr;
	
	int i = 0;
        for(int j = 0; j < 6; j++)
		for(int k = 0; k < 5; k++)
		        for(int l = 0; l < 4; l++, i++)
				memory[j][k][l] = i;

	struct aml_layout *a, *b, *c;
	
	assert(aml_layout_dense_create(&a,
				       (void *) memory,
				       AML_LAYOUT_ORDER_COLUMN_MAJOR,
				       sizeof(int),
				       3,
				       dims_col,
				       stride,
				       pitch_col) == AML_SUCCESS);
	
	assert(aml_layout_reshape(a, &b, 5, new_dims_col) == AML_SUCCESS);

	aml_layout_reshape_create(&c,
				  a,
				  AML_LAYOUT_ORDER_COLUMN_MAJOR,
				  5,
				  new_dims_col);

	i = 0;
	for(size_t j = 0; j < 3; j++)
	for(size_t k = 0; k < 2; k++)
	for(size_t l = 0; l < 5; l++)
	for(size_t m = 0; m < 2; m++)
	for(size_t n = 0; n < 2; n++, i++) {
		coords[0] = n;
		coords[1] = m;
		coords[2] = l;
		coords[3] = k;
		coords[4] = j;
		ptr = aml_layout_deref_safe(b, coords);
		assert(i == *(int *)ptr);
		ptr = aml_layout_deref_safe(c, coords);
		assert(i == *(int *)ptr);
	}

	free(a);
	free(b);
	free(c);

	assert(aml_layout_dense_create(&a,
				       (void *) memory,
				       AML_LAYOUT_ORDER_ROW_MAJOR,
				       sizeof(int),
				       3,
				       dims_row,
				       stride,
				       pitch_row) == AML_SUCCESS);
	
	assert(aml_layout_reshape(a, &b, 5, new_dims_row) == AML_SUCCESS);

	aml_layout_reshape_create(&c,
				  a,
				  AML_LAYOUT_ORDER_ROW_MAJOR,
				  5,
				  new_dims_row);

	i = 0;
	for(size_t j = 0; j < 3; j++)
	for(size_t k = 0; k < 2; k++)
	for(size_t l = 0; l < 5; l++)
	for(size_t m = 0; m < 2; m++)
	for(size_t n = 0; n < 2; n++, i++) {
		coords[0] = j;
		coords[1] = k;
		coords[2] = l;
		coords[3] = m;
		coords[4] = n;
		ptr = aml_layout_deref_safe(b, coords);
		assert(i == *(int *)ptr);
		ptr = aml_layout_deref_safe(c, coords);
		assert(i == *(int *)ptr);
	}

	free(a);
	free(b);
	free(c);
}

void test_reshape_strided(void)
{
	int memory[12][5][8];

	size_t dims_col[3] = {4, 5, 6};
	size_t dims_row[3] = {6, 5, 4};

	size_t stride[3] = {2, 1, 2};

	size_t pitch_col[3] = {8, 5, 12};
	size_t pitch_row[3] = {12, 5, 8};

	size_t new_dims_col[4] = {2, 10, 2, 3};
	size_t new_dims_row[4] = {3, 2, 10, 2};

	size_t coords[4];
	void *ptr;
	
	int i = 0;
	for(int j = 0; j < 6; j++)
		for(int k = 0; k < 5; k++)
			for(int l = 0; l < 4; l++, i++)
				memory[2*j][1*k][2*l] = i;

	struct aml_layout *a, *b, *c;
	
	assert(aml_layout_dense_create(&a,
				       (void *) memory,
				       AML_LAYOUT_ORDER_COLUMN_MAJOR,
				       sizeof(int),
				       3,
				       dims_col,
				       stride,
				       pitch_col) == AML_SUCCESS);
	
	assert(aml_layout_reshape(a, &b, 4, new_dims_col) == AML_SUCCESS);

	aml_layout_reshape_create(&c,
				  a,
				  AML_LAYOUT_ORDER_COLUMN_MAJOR,
				  4,
				  new_dims_col);

	i = 0;
	for(size_t j = 0; j < 3; j++)
	for(size_t k = 0; k < 2; k++)
	for(size_t l = 0; l < 10; l++)
	for(size_t m = 0; m < 2; m++, i++) {
		coords[0] = m;
		coords[1] = l;
		coords[2] = k;
		coords[3] = j;
		ptr = aml_layout_deref_safe(b, coords);
		assert(i == *(int *)ptr);
		ptr = aml_layout_deref_safe(c, coords);
		assert(i == *(int *)ptr);
	}

	free(a);
	free(b);
	free(c);

	assert(aml_layout_dense_create(&a,
				       (void *) memory,
				       AML_LAYOUT_ORDER_ROW_MAJOR,
				       sizeof(int),
				       3,
				       dims_row,
				       stride,
				       pitch_row) == AML_SUCCESS);
	
	assert(aml_layout_reshape(a, &b, 4, new_dims_row) == AML_SUCCESS);

	aml_layout_reshape_create(&c,
				  a,
				  AML_LAYOUT_ORDER_ROW_MAJOR,
				  4,
				  new_dims_row);

	i = 0;
	for(size_t j = 0; j < 3; j++)
	for(size_t k = 0; k < 2; k++)
	for(size_t l = 0; l < 10; l++)
	for(size_t m = 0; m < 2; m++, i++) {
		coords[0] = j;
		coords[1] = k;
		coords[2] = l;
		coords[3] = m;
		ptr = aml_layout_deref_safe(b, coords);
		assert(i == *(int *)ptr);
		ptr = aml_layout_deref_safe(b, coords);
		assert(i == *(int *)ptr);
	}

	free(a);
	free(b);
	free(c);
}

void test_base(void)
{
	struct aml_layout *a, *b;

	/* padd the dims to the closest multiple of 2 */
	float memory[16][12][8][8][4];
	size_t pitch[5] = {4, 8, 8, 12, 16};
	size_t cpitch[6] = {4, 4*4, 4*4*8, 4*4*8*8, 4*4*8*8*12, 4*4*8*8*12*16};
	size_t dims[5] = {2, 3, 7, 11, 13};
	size_t stride[5] = {1, 2, 1, 1, 1};

	size_t dims_col[5] = {2, 3, 7, 11, 13};
        size_t dims_row[5] = {13, 11, 7, 3, 2};

	size_t pitch_col[5] = {4, 8, 8, 12, 16};
	size_t pitch_row[5] = {16, 12, 8, 8, 4};

	size_t stride_col[5] = {1, 2, 1, 1, 1};
	size_t stride_row[5] = {1, 1, 1, 2, 1};

        for(size_t i = 0; i < 4*8*8*12*16; i++)
		((float*)(&memory[0][0][0][0][0]))[i] = (float)i;


	/* initialize column order layouts */
	assert(aml_layout_dense_create(&a,
				       (void *) memory,
				       AML_LAYOUT_ORDER_COLUMN_MAJOR,
				       sizeof(int),
				       5,
				       dims_col,
				       stride_col,
				       pitch_col) == AML_SUCCESS);
	assert(aml_layout_dense_create(&b,
				       (void *) memory,
				       AML_LAYOUT_ORDER_COLUMN_MAJOR,
				       sizeof(int),
				       5,
				       dims_col,
				       stride_col,
				       pitch_col) == AML_SUCCESS);
	
	struct aml_layout_dense *adataptr;
	struct aml_layout_dense *bdataptr;

	adataptr = (struct aml_layout_dense *)a->data;
	bdataptr = (struct aml_layout_dense *)b->data;
	assert( (intptr_t)(adataptr->stride) - (intptr_t)(adataptr->dims)
                == 5*sizeof(size_t) );
	assert( (intptr_t)(adataptr->pitch) - (intptr_t)(adataptr->dims)
                == 10*sizeof(size_t) );
	assert( (intptr_t)(adataptr->cpitch) - (intptr_t)(adataptr->dims)
                == 15*sizeof(size_t) );

	/* some simple checks */
	assert(!memcmp(adataptr->dims, dims, sizeof(size_t)*5));
	assert(!memcmp(adataptr->stride, stride, sizeof(size_t)*5));
	assert(!memcmp(adataptr->pitch, pitch, sizeof(size_t)*5));
	assert(!memcmp(adataptr->cpitch, cpitch, sizeof(size_t)*6));
	assert(!memcmp(bdataptr->dims, dims, sizeof(size_t)*5));
	assert(!memcmp(bdataptr->stride, stride, sizeof(size_t)*5));
	assert(!memcmp(bdataptr->pitch, pitch, sizeof(size_t)*5));
	assert(!memcmp(bdataptr->cpitch, cpitch, sizeof(size_t)*6));

	/* test column major subroutines */
	size_t dims_res[5];
	size_t coords_test_col[5] = { 1, 2, 3, 4, 5 };
	void *test_addr;
	void *res_addr = (void *)&memory[5][4][3][2*2][1];

	aml_layout_dims(a, dims_res);
	assert(!memcmp(dims_res, dims_col, sizeof(size_t)*5));
	test_addr = aml_layout_deref(a, coords_test_col);
	assert(res_addr == test_addr);
	assert(AML_LAYOUT_ORDER_COLUMN_MAJOR == aml_layout_order(a));

	free(a);

	/* initialize row order layouts */
	aml_layout_dense_create(&a,
				(void *)memory,
				AML_LAYOUT_ORDER_ROW_MAJOR,
				sizeof(float),
				5, dims_row,
				stride_row,
				pitch_row);
	
	adataptr = (struct aml_layout_dense *)a->data;
	bdataptr = (struct aml_layout_dense *)b->data;
	assert( (intptr_t)(adataptr->stride) - (intptr_t)(adataptr->dims)
                == 5*sizeof(size_t) );
	assert( (intptr_t)(adataptr->pitch) - (intptr_t)(adataptr->dims)
                == 10*sizeof(size_t) );
	assert( (intptr_t)(adataptr->cpitch) - (intptr_t)(adataptr->dims)
                == 15*sizeof(size_t) );

	/* some simple checks */
	assert(!memcmp(adataptr->dims, dims, sizeof(size_t)*5));
	assert(!memcmp(adataptr->stride, stride, sizeof(size_t)*5));
	assert(!memcmp(adataptr->pitch, pitch, sizeof(size_t)*5));
	assert(!memcmp(adataptr->cpitch, cpitch, sizeof(size_t)*6));
	assert(!memcmp(bdataptr->dims, dims, sizeof(size_t)*5));
	assert(!memcmp(bdataptr->stride, stride, sizeof(size_t)*5));
	assert(!memcmp(bdataptr->pitch, pitch, sizeof(size_t)*5));
	assert(!memcmp(bdataptr->cpitch, cpitch, sizeof(size_t)*6));

	/* test row major subroutines */
	size_t coords_test_row[5] = { 5, 4, 3, 2, 1 };
	aml_layout_dims(a, dims_res);
	assert(!memcmp(dims_res, dims_row, sizeof(size_t)*5));
	test_addr = aml_layout_deref(a, coords_test_row);
	assert(res_addr == test_addr);
	assert(AML_LAYOUT_ORDER_ROW_MAJOR == aml_layout_order(a));

	free(a);
}

int main(int argc, char *argv[])
{
	/* library initialization */
	aml_init(&argc, &argv);

	test_base();
	test_reshape_contiguous();
	test_reshape_discontiguous();
	test_reshape_strided();

	test_slice_contiguous();
	test_slice_strided();

	aml_finalize();
	return 0;
}

