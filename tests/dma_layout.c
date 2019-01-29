#include <aml.h>
#include <assert.h>

void test_dma_copy_generic()
{
	size_t elem_number[3] = { 5, 3, 2 };
	size_t c_src_pitch[3] = { 10, 6, 4 };
	size_t src_stride[3] = { 1, 1, 1};
	size_t c_dst_pitch[3] = { 5, 3, 2 };
	size_t dst_stride[3] = { 1, 1, 1};

	double src[4][6][10];
	double dst[2][3][5];

	double ref_dst[2][3][5];

	AML_LAYOUT_NATIVE_DECL(src_layout, 3);
	AML_LAYOUT_NATIVE_DECL(dst_layout, 3);
	AML_DMA_LAYOUT_DECL(dma);

	/* library initialization */

	aml_layout_native_ainit(&src_layout, AML_TYPE_LAYOUT_COLUMN_ORDER,
				(void *)src, sizeof(double), 3, elem_number,
				src_stride, c_src_pitch);
	aml_layout_native_ainit(&dst_layout, AML_TYPE_LAYOUT_COLUMN_ORDER,
				(void *)dst, sizeof(double), 3, elem_number,
				dst_stride, c_dst_pitch);
	aml_dma_layout_init(&dma, 1, aml_copy_layout_generic, NULL);

	for (int k = 0; k < 4; k++)
		for (int j = 0; j < 6; j++)
			for (int i = 0; i < 10; i++) {
				src[k][j][i] =
				    (double)(i + j * 10 + k * 10 * 6);
			}
	for (int k = 0; k < 2; k++)
		for (int j = 0; j < 3; j++)
			for (int i = 0; i < 5; i++) {
				dst[k][j][i] = 0.0;
				ref_dst[k][j][i] = src[k][j][i];
			}

	aml_dma_copy(&dma, &dst_layout, &src_layout);
	for (int k = 0; k < 2; k++)
		for (int j = 0; j < 3; j++)
			for (int i = 0; i < 5; i++)
				assert(ref_dst[k][j][i] == dst[k][j][i]);
	
	aml_dma_layout_destroy(&dma);
}

void test_dma_transpose_generic(void)
{
	size_t elem_number[4] = { 5, 3, 2, 4 };
	size_t elem_number2[4] = { 3, 2, 4, 5 };
	size_t c_src_pitch[4] = { 10, 6, 4, 8 };
	size_t src_stride[4] = { 2, 2, 2, 2 };
	size_t c_dst_pitch[4] = { 3, 2, 4, 5 };
	size_t dst_stride[4] = { 1, 1, 1, 1 };

	double src[8][4][6][10];
	double dst[5][4][2][3];

	double ref_dst[5][4][2][3];

	AML_LAYOUT_NATIVE_DECL(src_layout, 4);
	AML_LAYOUT_NATIVE_DECL(dst_layout, 4);
	AML_DMA_LAYOUT_DECL(dma);

	aml_layout_native_ainit(&src_layout, AML_TYPE_LAYOUT_COLUMN_ORDER,
				(void *)src, sizeof(double), 4, elem_number,
				src_stride, c_src_pitch);
	aml_layout_native_ainit(&dst_layout, AML_TYPE_LAYOUT_COLUMN_ORDER,
				(void *)dst, sizeof(double), 4, elem_number2,
				dst_stride, c_dst_pitch);
	aml_dma_layout_init(&dma, 1, aml_copy_layout_transpose_generic, NULL);

	for (int l = 0; l < 8; l++)
		for (int k = 0; k < 4; k++)
			for (int j = 0; j < 6; j++)
				for (int i = 0; i < 10; i++) {
					src[l][k][j][i] =
					    (double)(i + j * 10 + k * 10 * 6 +
						     l * 10 * 6 * 4);
				}
	for (int l = 0; l < 4; l++)
		for (int k = 0; k < 2; k++)
			for (int j = 0; j < 3; j++)
				for (int i = 0; i < 5; i++) {
					dst[i][l][k][j] = 0.0;
					ref_dst[i][l][k][j] =
					    src[2 * l][2 * k][2 * j][2 * i];
				}
	aml_dma_copy(&dma, &dst_layout, &src_layout);
	for (int l = 0; l < 4; l++)
		for (int k = 0; k < 2; k++)
			for (int j = 0; j < 3; j++)
				for (int i = 0; i < 5; i++)
					assert(ref_dst[i][l][k][j] ==
					       dst[i][l][k][j]);
	aml_dma_layout_destroy(&dma);
}

int main(int argc, char *argv[])
{
	aml_init(&argc, &argv);
	test_dma_copy_generic();
	aml_finalize();
	return 0;
}
