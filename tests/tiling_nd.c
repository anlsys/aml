#include <aml.h>
#include <assert.h>

void test_tiling_even(void)
{
	int memory[9][10][8];
	int memoryres[9][10][8];
	size_t dims_col[3] = {8, 10, 9};
	size_t dims_row[3] = {9, 10, 8};

	size_t stride[3] = {1, 1, 1};

	size_t dims_tile_col[3] = {4, 10, 3};
	size_t dims_tile_row[3] = {3, 10, 4};

	size_t expected_dims_col[3] = {2, 1, 3};
	size_t expected_dims_row[3] = {3, 1, 2};

        int l = 0;
	for(size_t i = 0; i < 9; i++)
	for(size_t j = 0; j < 10; j++)
	for(size_t k = 0; k < 8; k++, l++) {
		memory[i][j][k] = l;
		memoryres[i][j][k] = 0.0;
	}

	struct aml_layout *a, *ares;
	aml_layout_native_acreate(&a, AML_TYPE_LAYOUT_COLUMN_ORDER,
				  (void *)memory, sizeof(int), 3, dims_col,
				  stride, dims_col);
	aml_layout_native_acreate(&ares, AML_TYPE_LAYOUT_COLUMN_ORDER,
				  (void *)memoryres, sizeof(int), 3, dims_col,
				  stride, dims_col);


	struct aml_tiling_nd *t, *tres;
	aml_tiling_nd_resize_acreate(&t, AML_TYPE_TILING_COLUMN_ORDER,
				     a, 3, dims_tile_col);
	aml_tiling_nd_resize_acreate(&tres, AML_TYPE_TILING_COLUMN_ORDER,
				     ares, 3, dims_tile_col);


	assert(aml_tiling_nd_order(t) == AML_TYPE_TILING_COLUMN_ORDER);
	assert(aml_tiling_nd_ndims(t) == 3);

        size_t dims[3];
	aml_tiling_nd_tile_adims(t, dims);
	assert(memcmp(dims, dims_tile_col, 3*sizeof(size_t)) == 0);
	aml_tiling_nd_adims(t, dims);
	assert(memcmp(dims, expected_dims_col, 3*sizeof(size_t)) == 0);

	for(size_t i = 0; i < expected_dims_col[2]; i++)
	for(size_t j = 0; j < expected_dims_col[1]; j++)
	for(size_t k = 0; k < expected_dims_col[0]; k++) {
		struct aml_layout *b, *bres;
		b = aml_tiling_nd_index(t, k, j, i);
		bres = aml_tiling_nd_index(tres, k, j, i);
		aml_copy_layout_generic(bres, b);
		free(b);
		free(bres);
	}
	assert(memcmp(memory, memoryres, 8 * 10 * 9 *sizeof(int)) == 0);

	free(a);
	free(ares);
	free(t);
	free(tres);

	aml_layout_native_acreate(&a, AML_TYPE_LAYOUT_ROW_ORDER,
				  (void *)memory, sizeof(int), 3, dims_row,
				  stride, dims_row);
	aml_layout_native_acreate(&ares, AML_TYPE_LAYOUT_ROW_ORDER,
				  (void *)memoryres, sizeof(int), 3, dims_row,
				  stride, dims_row);


	aml_tiling_nd_resize_acreate(&t, AML_TYPE_TILING_ROW_ORDER,
				     a, 3, dims_tile_row);
	aml_tiling_nd_resize_acreate(&tres, AML_TYPE_TILING_ROW_ORDER,
				     ares, 3, dims_tile_row);

	assert(aml_tiling_nd_order(t) == AML_TYPE_TILING_ROW_ORDER);
	assert(aml_tiling_nd_ndims(t) == 3);

	aml_tiling_nd_tile_adims(t, dims);
	assert(memcmp(dims, dims_tile_row, 3*sizeof(size_t)) == 0);
	aml_tiling_nd_adims(t, dims);
	assert(memcmp(dims, expected_dims_row, 3*sizeof(size_t)) == 0);

	for(size_t i = 0; i < 9; i++)
	for(size_t j = 0; j < 10; j++)
	for(size_t k = 0; k < 8; k++, l++)
		memoryres[i][j][k] = 0.0;

	for(size_t i = 0; i < expected_dims_col[2]; i++)
	for(size_t j = 0; j < expected_dims_col[1]; j++)
	for(size_t k = 0; k < expected_dims_col[0]; k++) {
		struct aml_layout *b, *bres;
		b = aml_tiling_nd_index(t, i, j, k);
		bres = aml_tiling_nd_index(tres, i, j, k);
		aml_copy_layout_generic(bres, b);
		free(b);
		free(bres);
	}
	assert(memcmp(memory, memoryres, 8 * 10 * 9 *sizeof(int)) == 0);

	free(a);
	free(ares);
	free(t);
	free(tres);

}

void test_tiling_uneven(void)
{

	int memory[8][10][7];
	int memoryres[8][10][7];
	size_t dims_col[3] = {7, 10, 8};
	size_t dims_row[3] = {8, 10, 7};

	size_t stride[3] = {1, 1, 1};

	size_t dims_tile_col[3] = {4, 10, 3};
	size_t dims_tile_row[3] = {3, 10, 4};

	size_t expected_dims_col[3] = {2, 1, 3};
	size_t expected_dims_row[3] = {3, 1, 2};

        int l = 0;
	for(size_t i = 0; i < 8; i++)
	for(size_t j = 0; j < 10; j++)
	for(size_t k = 0; k < 7; k++, l++) {
		memory[i][j][k] = l;
		memoryres[i][j][k] = 0.0;
	}

	struct aml_layout *a, *ares;
	aml_layout_native_acreate(&a, AML_TYPE_LAYOUT_COLUMN_ORDER,
				  (void *)memory, sizeof(int), 3, dims_col,
				  stride, dims_col);
	aml_layout_native_acreate(&ares, AML_TYPE_LAYOUT_COLUMN_ORDER,
				  (void *)memoryres, sizeof(int), 3, dims_col,
				  stride, dims_col);


	struct aml_tiling_nd *t, *tres;
	aml_tiling_nd_resize_acreate(&t, AML_TYPE_TILING_COLUMN_ORDER,
				     a, 3, dims_tile_col);
	aml_tiling_nd_resize_acreate(&tres, AML_TYPE_TILING_COLUMN_ORDER,
				     ares, 3, dims_tile_col);


	assert(aml_tiling_nd_order(t) == AML_TYPE_TILING_COLUMN_ORDER);
	assert(aml_tiling_nd_ndims(t) == 3);

        size_t dims[3];
	aml_tiling_nd_tile_adims(t, dims);
	assert(memcmp(dims, dims_tile_col, 3*sizeof(size_t)) == 0);
	aml_tiling_nd_adims(t, dims);
	assert(memcmp(dims, expected_dims_col, 3*sizeof(size_t)) == 0);

	for(size_t i = 0; i < expected_dims_col[2]; i++)
	for(size_t j = 0; j < expected_dims_col[1]; j++)
	for(size_t k = 0; k < expected_dims_col[0]; k++) {
		struct aml_layout *b, *bres;
		b = aml_tiling_nd_index(t, k, j, i);
		bres = aml_tiling_nd_index(tres, k, j, i);
		aml_copy_layout_generic(bres, b);
		free(b);
		free(bres);
	}
	assert(memcmp(memory, memoryres, 7 * 10 * 8 *sizeof(int)) == 0);

	free(a);
	free(ares);
	free(t);
	free(tres);

	aml_layout_native_acreate(&a, AML_TYPE_LAYOUT_ROW_ORDER,
				  (void *)memory, sizeof(int), 3, dims_row,
				  stride, dims_row);
	aml_layout_native_acreate(&ares, AML_TYPE_LAYOUT_ROW_ORDER,
				  (void *)memoryres, sizeof(int), 3, dims_row,
				  stride, dims_row);

	aml_tiling_nd_resize_acreate(&t, AML_TYPE_TILING_ROW_ORDER,
				     a, 3, dims_tile_row);
	aml_tiling_nd_resize_acreate(&tres, AML_TYPE_TILING_ROW_ORDER,
				     ares, 3, dims_tile_row);

	assert(aml_tiling_nd_order(t) == AML_TYPE_TILING_ROW_ORDER);
	assert(aml_tiling_nd_ndims(t) == 3);

	aml_tiling_nd_tile_adims(t, dims);
	assert(memcmp(dims, dims_tile_row, 3*sizeof(size_t)) == 0);
	aml_tiling_nd_adims(t, dims);
	assert(memcmp(dims, expected_dims_row, 3*sizeof(size_t)) == 0);

	for(size_t i = 0; i < 8; i++)
	for(size_t j = 0; j < 10; j++)
	for(size_t k = 0; k < 7; k++, l++)
		memoryres[i][j][k] = 0.0;

	for(size_t i = 0; i < expected_dims_col[2]; i++)
	for(size_t j = 0; j < expected_dims_col[1]; j++)
	for(size_t k = 0; k < expected_dims_col[0]; k++) {
		struct aml_layout *b, *bres;
		b = aml_tiling_nd_index(t, i, j, k);
		bres = aml_tiling_nd_index(tres, i, j, k);
		aml_copy_layout_generic(bres, b);
		free(b);
		free(bres);
	}
	assert(memcmp(memory, memoryres, 7 * 10 * 8 *sizeof(int)) == 0);

	free(a);
	free(ares);
	free(t);
	free(tres);

}

void test_tiling_pad_even(void)
{
	int memory[9][10][8];
	int memoryres[9][10][8];
	size_t dims_col[3] = {8, 10, 9};
	size_t dims_row[3] = {9, 10, 8};

	size_t stride[3] = {1, 1, 1};

	size_t dims_tile_col[3] = {4, 10, 3};
	size_t dims_tile_row[3] = {3, 10, 4};

	size_t expected_dims_col[3] = {2, 1, 3};
	size_t expected_dims_row[3] = {3, 1, 2};

        int l = 0;
	for(size_t i = 0; i < 9; i++)
	for(size_t j = 0; j < 10; j++)
	for(size_t k = 0; k < 8; k++, l++) {
		memory[i][j][k] = l;
		memoryres[i][j][k] = 0.0;
	}

	struct aml_layout *a, *ares;
	aml_layout_native_acreate(&a, AML_TYPE_LAYOUT_COLUMN_ORDER,
				  (void *)memory, sizeof(int), 3, dims_col,
				  stride, dims_col);
	aml_layout_native_acreate(&ares, AML_TYPE_LAYOUT_COLUMN_ORDER,
				  (void *)memoryres, sizeof(int), 3, dims_col,
				  stride, dims_col);


	struct aml_tiling_nd *t, *tres;
	int neutral = 0xdeadbeef;
	aml_tiling_nd_pad_acreate(&t, AML_TYPE_TILING_COLUMN_ORDER,
				  a, 3, dims_tile_col, &neutral);
	aml_tiling_nd_pad_acreate(&tres, AML_TYPE_TILING_COLUMN_ORDER,
				  ares, 3, dims_tile_col, &neutral);


	assert(aml_tiling_nd_order(t) == AML_TYPE_TILING_COLUMN_ORDER);
	assert(aml_tiling_nd_ndims(t) == 3);

        size_t dims[3];
	aml_tiling_nd_tile_adims(t, dims);
	assert(memcmp(dims, dims_tile_col, 3*sizeof(size_t)) == 0);
	aml_tiling_nd_adims(t, dims);
	assert(memcmp(dims, expected_dims_col, 3*sizeof(size_t)) == 0);

	for(size_t i = 0; i < expected_dims_col[2]; i++)
	for(size_t j = 0; j < expected_dims_col[1]; j++)
	for(size_t k = 0; k < expected_dims_col[0]; k++) {
		struct aml_layout *b, *bres;
		b = aml_tiling_nd_index(t, k, j, i);
		bres = aml_tiling_nd_index(tres, k, j, i);
		aml_copy_layout_generic(bres, b);
		free(b);
		free(bres);
	}
	assert(memcmp(memory, memoryres, 8 * 10 * 9 *sizeof(int)) == 0);

	free(a);
	free(ares);
	free(t);
	free(tres);

	aml_layout_native_acreate(&a, AML_TYPE_LAYOUT_ROW_ORDER,
				  (void *)memory, sizeof(int), 3, dims_row,
				  stride, dims_row);
	aml_layout_native_acreate(&ares, AML_TYPE_LAYOUT_ROW_ORDER,
				  (void *)memoryres, sizeof(int), 3, dims_row,
				  stride, dims_row);

	aml_tiling_nd_pad_acreate(&t, AML_TYPE_TILING_ROW_ORDER,
				  a, 3, dims_tile_row, &neutral);
	aml_tiling_nd_pad_acreate(&tres, AML_TYPE_TILING_ROW_ORDER,
				  ares, 3, dims_tile_row, &neutral);

	assert(aml_tiling_nd_order(t) == AML_TYPE_TILING_ROW_ORDER);
	assert(aml_tiling_nd_ndims(t) == 3);

	aml_tiling_nd_tile_adims(t, dims);
	assert(memcmp(dims, dims_tile_row, 3*sizeof(size_t)) == 0);
	aml_tiling_nd_adims(t, dims);
	assert(memcmp(dims, expected_dims_row, 3*sizeof(size_t)) == 0);

	for(size_t i = 0; i < 9; i++)
	for(size_t j = 0; j < 10; j++)
	for(size_t k = 0; k < 8; k++, l++)
		memoryres[i][j][k] = 0.0;

	for(size_t i = 0; i < expected_dims_col[2]; i++)
	for(size_t j = 0; j < expected_dims_col[1]; j++)
	for(size_t k = 0; k < expected_dims_col[0]; k++) {
		struct aml_layout *b, *bres;
		b = aml_tiling_nd_index(t, i, j, k);
		bres = aml_tiling_nd_index(tres, i, j, k);
		aml_copy_layout_generic(bres, b);
		free(b);
		free(bres);
	}
	assert(memcmp(memory, memoryres, 8 * 10 * 9 *sizeof(int)) == 0);

	free(a);
	free(ares);
	free(t);
	free(tres);

}

void test_tiling_pad_uneven(void)
{

	int memory[8][10][7];
	int memoryres[9][10][8];
	size_t dims_col[3] = {7, 10, 8};
	size_t dims_row[3] = {8, 10, 7};
	size_t dims_col_res[3] = {8, 10, 9};
	size_t dims_row_res[3] = {9, 10, 8};

	size_t stride[3] = {1, 1, 1};

	size_t dims_tile_col[3] = {4, 10, 3};
	size_t dims_tile_row[3] = {3, 10, 4};

	size_t expected_dims_col[3] = {2, 1, 3};
	size_t expected_dims_row[3] = {3, 1, 2};

        int l = 0;
	for(size_t i = 0; i < 8; i++)
	for(size_t j = 0; j < 10; j++)
	for(size_t k = 0; k < 7; k++, l++)
		memory[i][j][k] = l;

	for(size_t i = 0; i < 9; i++)
	for(size_t j = 0; j < 10; j++)
	for(size_t k = 0; k < 8; k++, l++)
		memoryres[i][j][k] = 0.0;


	struct aml_layout *a, *ares;
	aml_layout_native_acreate(&a, AML_TYPE_LAYOUT_COLUMN_ORDER,
				  (void *)memory, sizeof(int), 3, dims_col,
				  stride, dims_col);
	aml_layout_native_acreate(&ares, AML_TYPE_LAYOUT_COLUMN_ORDER,
				  (void *)memoryres, sizeof(int), 3, dims_col_res,
				  stride, dims_col_res);


	struct aml_tiling_nd *t, *tres;
	int neutral = 0xdeadbeef;
	aml_tiling_nd_pad_acreate(&t, AML_TYPE_TILING_COLUMN_ORDER,
				  a, 3, dims_tile_col, &neutral);
	aml_tiling_nd_pad_acreate(&tres, AML_TYPE_TILING_COLUMN_ORDER,
				  ares, 3, dims_tile_col, &neutral);


	assert(aml_tiling_nd_order(t) == AML_TYPE_TILING_COLUMN_ORDER);
	assert(aml_tiling_nd_ndims(t) == 3);

        size_t dims[3];
	aml_tiling_nd_tile_adims(t, dims);
	assert(memcmp(dims, dims_tile_col, 3*sizeof(size_t)) == 0);
	aml_tiling_nd_adims(t, dims);
	assert(memcmp(dims, expected_dims_col, 3*sizeof(size_t)) == 0);

	for(size_t i = 0; i < expected_dims_col[2]; i++)
	for(size_t j = 0; j < expected_dims_col[1]; j++)
	for(size_t k = 0; k < expected_dims_col[0]; k++) {
		struct aml_layout *b, *bres;
		b = aml_tiling_nd_index(t, k, j, i);
		bres = aml_tiling_nd_index(tres, k, j, i);
		aml_copy_layout_generic(bres, b);
		free(b);
		free(bres);
	}
	for(size_t i = 0; i < 9; i++)
	for(size_t j = 0; j < 10; j++)
	for(size_t k = 0; k < 8; k++, l++)
		if( k >= 7 || i >= 8)
			assert(memoryres[i][j][k] == 0xdeadbeef);
		else
			assert(memoryres[i][j][k] == memory[i][j][k]);

	free(a);
	free(t);

	aml_layout_native_acreate(&a, AML_TYPE_LAYOUT_ROW_ORDER,
				  (void *)memory, sizeof(int), 3, dims_row,
				  stride, dims_row);
	aml_layout_native_acreate(&ares, AML_TYPE_LAYOUT_ROW_ORDER,
				  (void *)memoryres, sizeof(int), 3, dims_row_res,
				  stride, dims_row_res);

	aml_tiling_nd_pad_acreate(&t, AML_TYPE_TILING_ROW_ORDER,
				  a, 3, dims_tile_row, &neutral);
	aml_tiling_nd_pad_acreate(&tres, AML_TYPE_TILING_ROW_ORDER,
				  ares, 3, dims_tile_row, &neutral);

	assert(aml_tiling_nd_order(t) == AML_TYPE_TILING_ROW_ORDER);
	assert(aml_tiling_nd_ndims(t) == 3);

	aml_tiling_nd_tile_adims(t, dims);
	assert(memcmp(dims, dims_tile_row, 3*sizeof(size_t)) == 0);
	aml_tiling_nd_adims(t, dims);
	assert(memcmp(dims, expected_dims_row, 3*sizeof(size_t)) == 0);

	for(size_t i = 0; i < 9; i++)
	for(size_t j = 0; j < 10; j++)
	for(size_t k = 0; k < 8; k++, l++)
		memoryres[i][j][k] = 0.0;

	for(size_t i = 0; i < expected_dims_col[2]; i++)
	for(size_t j = 0; j < expected_dims_col[1]; j++)
	for(size_t k = 0; k < expected_dims_col[0]; k++) {
		struct aml_layout *b, *bres;
		b = aml_tiling_nd_index(t, i, j, k);
		bres = aml_tiling_nd_index(tres, i, j, k);
		aml_copy_layout_generic(bres, b);
		free(b);
		free(bres);
	}
	for(size_t i = 0; i < 9; i++)
	for(size_t j = 0; j < 10; j++)
	for(size_t k = 0; k < 8; k++, l++)
		if( k >= 7 || i >= 8)
			assert(memoryres[i][j][k] == 0xdeadbeef);
		else
			assert(memoryres[i][j][k] == memory[i][j][k]);

	free(a);
	free(ares);
	free(t);
	free(tres);

}

int main(int argc, char *argv[])
{
	/* library initialization */
	aml_init(&argc, &argv);

	test_tiling_even();
	test_tiling_uneven();
	test_tiling_pad_even();
	test_tiling_pad_uneven();

	return 0;
}

