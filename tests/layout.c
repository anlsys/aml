#include <aml.h>
#include <assert.h>

int main(int argc, char *argv[])
{
	struct aml_layout *a;
	AML_LAYOUT_DECL(b, 5);

	/* padd the dims to the closest multiple of 2 */
	float memory[16][12][8][8][4];
	size_t cpitch[5] = {4, 4*4, 4*4*8, 4*4*8*8, 4*4*8*8*12};
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

	/* library initialization */
	aml_init(&argc, &argv);

	/* initialize column order layouts */
	aml_layout_acreate(&a, AML_TYPE_LAYOUT_COLUMN_ORDER, (void *)memory,
			   sizeof(float), 5, dims_col, stride_col, pitch_col);
	aml_layout_ainit(&b, AML_TYPE_LAYOUT_COLUMN_ORDER, (void *)memory,
			 sizeof(float), 5, dims_col, stride_col, pitch_col);

	assert( (intptr_t)(a->data->stride) - (intptr_t)(a->data->dims)
                == 5*sizeof(size_t) );
	assert( (intptr_t)(a->data->pitch) - (intptr_t)(a->data->dims)
                == 10*sizeof(size_t) );

	/* some simple checks */
	assert(!memcmp(a->data->dims, dims, sizeof(size_t)*5));
	assert(!memcmp(a->data->pitch, cpitch, sizeof(size_t)*5));
	assert(!memcmp(a->data->stride, stride, sizeof(size_t)*5));
	assert(!memcmp(b.data->dims, dims, sizeof(size_t)*5));
	assert(!memcmp(b.data->pitch, cpitch, sizeof(size_t)*5));
	assert(!memcmp(b.data->stride, stride, sizeof(size_t)*5));

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
	aml_layout_acreate(&a, AML_TYPE_LAYOUT_ROW_ORDER, (void *)memory,
			   sizeof(float), 5, dims_row, stride_row, pitch_row);
	aml_layout_ainit(&b, AML_TYPE_LAYOUT_ROW_ORDER, (void *)memory,
			 sizeof(float), 5, dims_row, stride_row, pitch_row);

	assert( (intptr_t)(a->data->stride) - (intptr_t)(a->data->dims)
                == 5*sizeof(size_t) );
	assert( (intptr_t)(a->data->pitch) - (intptr_t)(a->data->dims)
                == 10*sizeof(size_t) );

	/* some simple checks */
	assert(!memcmp(a->data->dims, dims, sizeof(size_t)*5));
	assert(!memcmp(a->data->pitch, cpitch, sizeof(size_t)*5));
	assert(!memcmp(a->data->stride, stride, sizeof(size_t)*5));
	assert(!memcmp(b.data->dims, dims, sizeof(size_t)*5));
	assert(!memcmp(b.data->pitch, cpitch, sizeof(size_t)*5));
	assert(!memcmp(b.data->stride, stride, sizeof(size_t)*5));

	/* test column major subroutines */
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

	aml_finalize();
	return 0;
}
