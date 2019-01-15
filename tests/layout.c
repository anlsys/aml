#include <aml.h>
#include <assert.h>


void test_reshape_contiguous(void)
{
	int memory[4*5*6];

	size_t dims_col[3] = {4, 5, 6};
	size_t dims_row[3] = {6, 5, 4};

	size_t stride[3] = {1, 1, 1};

	size_t new_dims_col[2] = {24, 5};
	size_t new_dims_row[2] = {5, 24};

	int i;
        for(i = 0; i < 4*5*6; i++)
		memory[i] = i;

	struct aml_layout *a;
	aml_layout_native_acreate(&a, AML_TYPE_LAYOUT_COLUMN_ORDER,
				  (void *)memory, sizeof(int), 3, dims_col,
				  stride, dims_col);
	struct aml_layout *b = aml_layout_areshape(a, 2, new_dims_col);
	assert(AML_TYPE_LAYOUT_COLUMN_ORDER == aml_layout_order(b));

	i = 0;
	for(size_t j = 0; j < 5; j++)
		for(size_t k = 0; k < 24; k++, i++)
			assert(i == *(int *)aml_layout_deref(b, k, j));

	free(a);
	free(b);

	aml_layout_native_acreate(&a, AML_TYPE_LAYOUT_ROW_ORDER,
				  (void *)memory, sizeof(int), 3, dims_row,
				  stride, dims_row);
	b = aml_layout_areshape(a, 2, new_dims_row);
	assert(AML_TYPE_LAYOUT_ROW_ORDER == aml_layout_order(b));

	i = 0;
	for(size_t j = 0; j < 5; j++)
		for(size_t k = 0; k < 24; k++, i++)
			assert(i == *(int *)aml_layout_deref(b, j, k));

	free(a);
	free(b);
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

	int i = 0;
        for(int j = 0; j < 6; j++)
		for(int k = 0; k < 5; k++)
		        for(int l = 0; l < 4; l++, i++)
				memory[j][k][l] = i;

	struct aml_layout *a;
	aml_layout_native_acreate(&a, AML_TYPE_LAYOUT_COLUMN_ORDER,
				  (void *)memory, sizeof(int), 3, dims_col,
				  stride, pitch_col);
	struct aml_layout *b = aml_layout_areshape(a, 5, new_dims_col);

	i = 0;
	for(size_t j = 0; j < 3; j++)
	for(size_t k = 0; k < 2; k++)
	for(size_t l = 0; l < 5; l++)
	for(size_t m = 0; m < 2; m++)
	for(size_t n = 0; n < 2; n++, i++)
		assert(i == *(int *)aml_layout_deref(b, n, m, l, k, j));

	free(a);
	free(b);

	aml_layout_native_acreate(&a, AML_TYPE_LAYOUT_ROW_ORDER,
				  (void *)memory, sizeof(int), 3, dims_row,
				  stride, pitch_row);
	b = aml_layout_areshape(a, 5, new_dims_row);

	i = 0;
	for(size_t j = 0; j < 3; j++)
	for(size_t k = 0; k < 2; k++)
	for(size_t l = 0; l < 5; l++)
	for(size_t m = 0; m < 2; m++)
	for(size_t n = 0; n < 2; n++, i++)
		assert(i == *(int *)aml_layout_deref(b, j, k, l, m, n));

	free(a);
	free(b);
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

	int i = 0;
	for(int j = 0; j < 6; j++)
		for(int k = 0; k < 5; k++)
			for(int l = 0; l < 4; l++, i++)
				memory[2*j][1*k][2*l] = i;

	struct aml_layout *a;
	aml_layout_native_acreate(&a, AML_TYPE_LAYOUT_COLUMN_ORDER,
				  (void *)memory, sizeof(int), 3, dims_col,
				  stride, pitch_col);
	struct aml_layout *b = aml_layout_areshape(a, 4, new_dims_col);

	i = 0;
	for(size_t j = 0; j < 3; j++)
	for(size_t k = 0; k < 2; k++)
	for(size_t l = 0; l < 10; l++)
	for(size_t m = 0; m < 2; m++, i++)
		assert(i == *(int *)aml_layout_deref(b, m, l, k, j));

	free(a);
	free(b);

	aml_layout_native_acreate(&a, AML_TYPE_LAYOUT_ROW_ORDER,
				  (void *)memory, sizeof(int), 3, dims_row,
				  stride, pitch_row);
	b = aml_layout_areshape(a, 4, new_dims_row);

	i = 0;
	for(size_t j = 0; j < 3; j++)
	for(size_t k = 0; k < 2; k++)
	for(size_t l = 0; l < 10; l++)
	for(size_t m = 0; m < 2; m++, i++)
		assert(i == *(int *)aml_layout_deref(b, j, k, l, m));

	free(a);
	free(b);
}

void test_base(void)
{
	struct aml_layout *a;
	AML_LAYOUT_NATIVE_DECL(b, 5);

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
	aml_layout_native_acreate(&a, AML_TYPE_LAYOUT_COLUMN_ORDER,
				  (void *)memory, sizeof(float), 5, dims_col,
				  stride_col, pitch_col);
	aml_layout_native_ainit(&b, AML_TYPE_LAYOUT_COLUMN_ORDER,
				(void *)memory, sizeof(float), 5, dims_col,
				stride_col, pitch_col);

	struct aml_layout_data_native *adataptr;
	struct aml_layout_data_native *bdataptr;

	adataptr = (struct aml_layout_data_native *)a->data;
	bdataptr = (struct aml_layout_data_native *)b.data;
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

	aml_layout_adims(a, dims_res);
	assert(!memcmp(dims_res, dims_col, sizeof(size_t)*5));
	aml_layout_dims(a, dims_res,
			   dims_res + 1,
			   dims_res + 2,
			   dims_res + 3,
			   dims_res + 4);
	assert(!memcmp(dims_res, dims_col, sizeof(size_t)*5));
	test_addr = aml_layout_aderef(a, coords_test_col);
	assert(res_addr == test_addr);
	test_addr = aml_layout_deref(a, coords_test_col[0],
					coords_test_col[1],
					coords_test_col[2],
					coords_test_col[3],
					coords_test_col[4]);
	assert(res_addr == test_addr);
	assert(AML_TYPE_LAYOUT_COLUMN_ORDER == aml_layout_order(a));

	free(a);

	/* initialize row order layouts */
	aml_layout_native_acreate(&a, AML_TYPE_LAYOUT_ROW_ORDER, (void *)memory,
				  sizeof(float), 5, dims_row, stride_row,
				  pitch_row);
	aml_layout_native_ainit(&b, AML_TYPE_LAYOUT_ROW_ORDER, (void *)memory,
				sizeof(float), 5, dims_row, stride_row,
				pitch_row);

	adataptr = (struct aml_layout_data_native *)a->data;
	bdataptr = (struct aml_layout_data_native *)b.data;
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
	aml_layout_adims(a, dims_res);
	assert(!memcmp(dims_res, dims_row, sizeof(size_t)*5));
	aml_layout_dims(a, dims_res,
			   dims_res + 1,
			   dims_res + 2,
			   dims_res + 3,
			   dims_res + 4);
	assert(!memcmp(dims_res, dims_row, sizeof(size_t)*5));
	test_addr = aml_layout_aderef(a, coords_test_row);
	assert(res_addr == test_addr);
	test_addr = aml_layout_deref(a, coords_test_row[0],
					coords_test_row[1],
					coords_test_row[2],
					coords_test_row[3],
					coords_test_row[4]);
	assert(res_addr == test_addr);
	assert(AML_TYPE_LAYOUT_ROW_ORDER == aml_layout_order(a));

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

	aml_finalize();
	return 0;
}

