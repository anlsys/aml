#include <aml.h>
#include <assert.h>

void test_copy_2d(void)
{
	size_t elem_number[2] = { 5, 3 };
	size_t src_pitch[2] = { 10, 6 };
	size_t dst_pitch[2] = { 5, 3 };

	double src[6][10];
	double dst[3][5];
	double dst2[6][10];

	double ref_dst2[6][10];
	double ref_dst[3][5];

	for (int j = 0; j < 6; j++)
		for (int i = 0; i < 10; i++) {
			src[j][i] = (double)(i + j * 10);
			ref_dst2[j][i] = 0.0;
			dst2[j][i] = 0.0;
		}
	for (int j = 0; j < 3; j++)
		for (int i = 0; i < 5; i++) {
			dst[j][i] = 0.0;
			ref_dst[j][i] = src[j][i];
			ref_dst2[j][i] = src[j][i];
		}

	aml_copy_nd(2, dst, dst_pitch, src, src_pitch, elem_number,
		    sizeof(double));
	for (int j = 0; j < 3; j++)
		for (int i = 0; i < 5; i++)
			assert(ref_dst[j][i] == dst[j][i]);

	aml_copy_nd(2, dst2, src_pitch, dst, dst_pitch, elem_number,
		    sizeof(double));
	for (int j = 0; j < 6; j++)
		for (int i = 0; i < 10; i++)
			assert(ref_dst2[j][i] == dst2[j][i]);

}

void test_copy_t2d(void)
{
	size_t elem_number[2] = { 5, 3 };
	size_t elem_number2[2] = { 3, 5 };
	size_t src_pitch[2] = { 10, 6 };
	size_t dst_pitch[2] = { 3, 5 };

	double src[6][10];
	double dst[5][3];
	double dst2[6][10];

	double ref_dst2[6][10];
	double ref_dst[5][3];

	for (int j = 0; j < 6; j++)
		for (int i = 0; i < 10; i++) {
			src[j][i] = (double)(i + j * 10);
			ref_dst2[j][i] = 0.0;
			dst2[j][i] = 0.0;
		}
	for (int j = 0; j < 3; j++)
		for (int i = 0; i < 5; i++) {
			dst[i][j] = 0.0;
			ref_dst[i][j] = src[j][i];
			ref_dst2[j][i] = src[j][i];
		}

	aml_copy_tnd(2, dst, dst_pitch, src, src_pitch, elem_number,
		     sizeof(double));
	for (int j = 0; j < 3; j++)
		for (int i = 0; i < 5; i++)
			assert(ref_dst[i][j] == dst[i][j]);

	aml_copy_tnd(2, dst2, src_pitch, dst, dst_pitch, elem_number2,
		     sizeof(double));
	for (int j = 0; j < 6; j++)
		for (int i = 0; i < 10; i++)
			assert(ref_dst2[j][i] == dst2[j][i]);

}

void test_copy_3d(void)
{
	size_t elem_number[3] = { 5, 3, 2 };
	size_t src_pitch[3] = { 10, 6, 4 };
	size_t dst_pitch[3] = { 5, 3, 2 };

	double src[4][6][10];
	double dst[2][3][5];
	double dst2[4][6][10];

	double ref_dst2[4][6][10];
	double ref_dst[2][3][5];

	for (int k = 0; k < 4; k++)
		for (int j = 0; j < 6; j++)
			for (int i = 0; i < 10; i++) {
				src[k][j][i] =
				    (double)(i + j * 10 + k * 10 * 6);
				ref_dst2[k][j][i] = 0.0;
				dst2[k][j][i] = 0.0;
			}
	for (int k = 0; k < 2; k++)
		for (int j = 0; j < 3; j++)
			for (int i = 0; i < 5; i++) {
				dst[k][j][i] = 0.0;
				ref_dst[k][j][i] = src[k][j][i];
				ref_dst2[k][j][i] = src[k][j][i];
			}

	aml_copy_nd(3, dst, dst_pitch, src, src_pitch, elem_number,
		    sizeof(double));
	for (int k = 0; k < 2; k++)
		for (int j = 0; j < 3; j++)
			for (int i = 0; i < 5; i++)
				assert(ref_dst[k][j][i] == dst[k][j][i]);

	aml_copy_nd(3, dst2, src_pitch, dst, dst_pitch, elem_number,
		    sizeof(double));
	for (int k = 0; k < 4; k++)
		for (int j = 0; j < 6; j++)
			for (int i = 0; i < 10; i++)
				assert(ref_dst2[k][j][i] == dst2[k][j][i]);

}

void test_copy_3d_c(void)
{
	size_t elem_number[3] = { 5, 3, 2 };
	size_t c_src_pitch[4] = { 8, 8 * 10, 8 * 10 * 6, 8 * 10 * 6 * 4 };
	size_t c_dst_pitch[4] = { 8, 8 * 5, 8 * 5 * 3, 8 * 5 * 3 * 2 };

	double src[4][6][10];
	double dst[2][3][5];
	double dst2[4][6][10];

	double ref_dst2[4][6][10];
	double ref_dst[2][3][5];

	for (int k = 0; k < 4; k++)
		for (int j = 0; j < 6; j++)
			for (int i = 0; i < 10; i++) {
				src[k][j][i] =
				    (double)(i + j * 10 + k * 10 * 6);
				ref_dst2[k][j][i] = 0.0;
				dst2[k][j][i] = 0.0;
			}
	for (int k = 0; k < 2; k++)
		for (int j = 0; j < 3; j++)
			for (int i = 0; i < 5; i++) {
				dst[k][j][i] = 0.0;
				ref_dst[k][j][i] = src[k][j][i];
				ref_dst2[k][j][i] = src[k][j][i];
			}

	aml_copy_nd_c(3, dst, c_dst_pitch, src, c_src_pitch, elem_number,
		    sizeof(double));
	for (int k = 0; k < 2; k++)
		for (int j = 0; j < 3; j++)
			for (int i = 0; i < 5; i++)
				assert(ref_dst[k][j][i] == dst[k][j][i]);

	aml_copy_nd_c(3, dst2, c_src_pitch, dst, c_dst_pitch, elem_number,
		    sizeof(double));
	for (int k = 0; k < 4; k++)
		for (int j = 0; j < 6; j++)
			for (int i = 0; i < 10; i++)
				assert(ref_dst2[k][j][i] == dst2[k][j][i]);

}

void test_copy_3dstr(void)
{
	size_t elem_number[3] = { 5, 3, 2 };
	size_t src_pitch[3] = { 10, 6, 4 };
	size_t src_stride[3] = { 2, 2, 2 };
	size_t dst_pitch[3] = { 5, 3, 2 };
	size_t dst_stride[3] = { 1, 1, 1 };

	double src[4][6][10];
	double dst[2][3][5];
	double dst2[4][6][10];

	double ref_dst2[4][6][10];
	double ref_dst[2][3][5];

	for (int k = 0; k < 4; k++)
		for (int j = 0; j < 6; j++)
			for (int i = 0; i < 10; i++) {
				src[k][j][i] =
				    (double)(i + j * 10 + k * 10 * 6);
				ref_dst2[k][j][i] = 0.0;
				dst2[k][j][i] = 0.0;
			}
	for (int k = 0; k < 2; k++)
		for (int j = 0; j < 3; j++)
			for (int i = 0; i < 5; i++) {
				dst[k][j][i] = 0.0;
				ref_dst[k][j][i] = src[2 * k][2 * j][2 * i];
				ref_dst2[2 * k][2 * j][2 * i] =
				    src[2 * k][2 * j][2 * i];
			}

	aml_copy_ndstr(3, dst, dst_pitch, dst_stride, src, src_pitch,
		       src_stride, elem_number, sizeof(double));
	for (int k = 0; k < 2; k++)
		for (int j = 0; j < 3; j++)
			for (int i = 0; i < 5; i++)
				assert(ref_dst[k][j][i] == dst[k][j][i]);

	aml_copy_ndstr(3, dst2, src_pitch, src_stride, dst, dst_pitch,
		       dst_stride, elem_number, sizeof(double));
	for (int k = 0; k < 4; k++)
		for (int j = 0; j < 6; j++)
			for (int i = 0; i < 10; i++)
				assert(ref_dst2[k][j][i] == dst2[k][j][i]);
}

void test_copy_3dstr_c(void)
{
	size_t elem_number[3] = { 5, 3, 2 };
	size_t c_src_pitch[4] = { 8, 8 * 10, 8 * 10 * 6, 8 * 10 * 6 * 4 };
	size_t src_stride[3] = { 2, 2, 2 };
	size_t c_dst_pitch[4] = { 8, 8 * 5, 8 * 5 * 3, 8 * 5 * 3 * 2 };
	size_t dst_stride[3] = { 1, 1, 1 };

	double src[4][6][10];
	double dst[2][3][5];
	double dst2[4][6][10];

	double ref_dst2[4][6][10];
	double ref_dst[2][3][5];

	for (int k = 0; k < 4; k++)
		for (int j = 0; j < 6; j++)
			for (int i = 0; i < 10; i++) {
				src[k][j][i] =
				    (double)(i + j * 10 + k * 10 * 6);
				ref_dst2[k][j][i] = 0.0;
				dst2[k][j][i] = 0.0;
			}
	for (int k = 0; k < 2; k++)
		for (int j = 0; j < 3; j++)
			for (int i = 0; i < 5; i++) {
				dst[k][j][i] = 0.0;
				ref_dst[k][j][i] = src[2 * k][2 * j][2 * i];
				ref_dst2[2 * k][2 * j][2 * i] =
				    src[2 * k][2 * j][2 * i];
			}

	aml_copy_ndstr_c(3, dst, c_dst_pitch, dst_stride, src, c_src_pitch,
			 src_stride, elem_number, sizeof(double));
	for (int k = 0; k < 2; k++)
		for (int j = 0; j < 3; j++)
			for (int i = 0; i < 5; i++)
				assert(ref_dst[k][j][i] == dst[k][j][i]);

	aml_copy_ndstr_c(3, dst2, c_src_pitch, src_stride, dst, c_dst_pitch,
		       dst_stride, elem_number, sizeof(double));
	for (int k = 0; k < 4; k++)
		for (int j = 0; j < 6; j++)
			for (int i = 0; i < 10; i++)
				assert(ref_dst2[k][j][i] == dst2[k][j][i]);
}

void test_copy_t3d(void)
{
	size_t elem_number[3] = { 5, 3, 2 };
	size_t elem_number2[3] = { 3, 2, 5 };
	size_t elem_number3[3] = { 2, 5, 3 };
	size_t src_pitch[3] = { 10, 6, 4 };
	size_t dst_pitch[3] = { 3, 2, 5 };
	size_t dst_pitch2[3] = { 2, 5, 3 };

	double src[4][6][10];
	double dst[5][2][3];
	double dst2[3][5][2];
	double dst3[4][6][10];

	double ref_dst[5][2][3];
	double ref_dst2[3][5][2];
	double ref_dst3[4][6][10];

	for (int k = 0; k < 4; k++)
		for (int j = 0; j < 6; j++)
			for (int i = 0; i < 10; i++) {
				src[k][j][i] =
				    (double)(i + j * 10 + k * 10 * 6);
				ref_dst3[k][j][i] = 0.0;
				dst3[k][j][i] = 0.0;
			}
	for (int k = 0; k < 2; k++)
		for (int j = 0; j < 3; j++)
			for (int i = 0; i < 5; i++) {
				dst[i][k][j] = 0.0;
				dst2[j][i][k] = 0.0;
				ref_dst[i][k][j] = src[k][j][i];
				ref_dst2[j][i][k] = src[k][j][i];
				ref_dst3[k][j][i] = src[k][j][i];
			}

	aml_copy_tnd(3, dst, dst_pitch, src, src_pitch, elem_number,
		     sizeof(double));
	for (int k = 0; k < 2; k++)
		for (int j = 0; j < 3; j++)
			for (int i = 0; i < 5; i++)
				assert(ref_dst[i][k][j] == dst[i][k][j]);

	aml_copy_tnd(3, dst2, dst_pitch2, dst, dst_pitch, elem_number2,
		     sizeof(double));
	for (int k = 0; k < 2; k++)
		for (int j = 0; j < 3; j++)
			for (int i = 0; i < 5; i++)
				assert(ref_dst2[j][i][k] == dst2[j][i][k]);

	aml_copy_tnd(3, dst3, src_pitch, dst2, dst_pitch2, elem_number3,
		     sizeof(double));
	for (int k = 0; k < 4; k++)
		for (int j = 0; j < 6; j++)
			for (int i = 0; i < 10; i++)
				assert(ref_dst3[k][j][i] == dst3[k][j][i]);
}

void test_copy_rt3d(void)
{
	size_t elem_number[3] = { 5, 3, 2 };
	size_t elem_number2[3] = { 2, 5, 3 };
	size_t elem_number3[3] = { 3, 2, 5 };
	size_t src_pitch[3] = { 10, 6, 4 };
	size_t dst_pitch[3] = { 2, 5, 3 };
	size_t dst_pitch2[3] = { 3, 2, 5 };

	double src[4][6][10];
	double dst[3][5][2];
	double dst2[5][2][3];
	double dst3[4][6][10];

	double ref_dst[3][5][2];
	double ref_dst2[5][2][3];
	double ref_dst3[4][6][10];

	for (int k = 0; k < 4; k++)
		for (int j = 0; j < 6; j++)
			for (int i = 0; i < 10; i++) {
				src[k][j][i] =
				    (double)(i + j * 10 + k * 10 * 6);
				ref_dst3[k][j][i] = 0.0;
				dst3[k][j][i] = 0.0;
			}
	for (int k = 0; k < 2; k++)
		for (int j = 0; j < 3; j++)
			for (int i = 0; i < 5; i++) {
				dst[j][i][k] = 0.0;
				dst2[i][k][j] = 0.0;
				ref_dst[j][i][k] = src[k][j][i];
				ref_dst2[i][k][j] = src[k][j][i];
				ref_dst3[k][j][i] = src[k][j][i];
			}

	aml_copy_rtnd(3, dst, dst_pitch, src, src_pitch, elem_number,
		      sizeof(double));
	for (int k = 0; k < 2; k++)
		for (int j = 0; j < 3; j++)
			for (int i = 0; i < 5; i++)
				assert(ref_dst[j][i][k] == dst[j][i][k]);

	aml_copy_rtnd(3, dst2, dst_pitch2, dst, dst_pitch, elem_number2,
		      sizeof(double));
	for (int k = 0; k < 2; k++)
		for (int j = 0; j < 3; j++)
			for (int i = 0; i < 5; i++)
				assert(ref_dst2[i][k][j] == dst2[i][k][j]);

	aml_copy_rtnd(3, dst3, src_pitch, dst2, dst_pitch2, elem_number3,
		      sizeof(double));
	for (int k = 0; k < 4; k++)
		for (int j = 0; j < 6; j++)
			for (int i = 0; i < 10; i++)
				assert(ref_dst3[k][j][i] == dst3[k][j][i]);
}

void test_copy_t4d(void)
{
	size_t elem_number[4] = { 5, 3, 2, 4 };
	size_t elem_number2[4] = { 3, 2, 4, 5 };
	size_t src_pitch[4] = { 10, 6, 4, 8 };
	size_t dst_pitch[4] = { 3, 2, 4, 5 };

	double src[8][4][6][10];
	double dst[5][4][2][3];
	double dst2[8][4][6][10];

	double ref_dst[5][4][2][3];
	double ref_dst2[8][4][6][10];

	for (int l = 0; l < 8; l++)
		for (int k = 0; k < 4; k++)
			for (int j = 0; j < 6; j++)
				for (int i = 0; i < 10; i++) {
					src[l][k][j][i] =
					    (double)(i + j * 10 + k * 10 * 6 +
						     l * 10 * 6 * 4);
					ref_dst2[l][k][j][i] = 0.0;
					dst2[l][k][j][i] = 0.0;
				}
	for (int l = 0; l < 4; l++)
		for (int k = 0; k < 2; k++)
			for (int j = 0; j < 3; j++)
				for (int i = 0; i < 5; i++) {
					dst[i][l][k][j] = 0.0;
					ref_dst[i][l][k][j] = src[l][k][j][i];
					ref_dst2[l][k][j][i] = src[l][k][j][i];
				}

	aml_copy_tnd(4, dst, dst_pitch, src, src_pitch, elem_number,
		     sizeof(double));
	for (int l = 0; l < 4; l++)
		for (int k = 0; k < 2; k++)
			for (int j = 0; j < 3; j++)
				for (int i = 0; i < 5; i++)
					assert(ref_dst[i][l][k][j] ==
					       dst[i][l][k][j]);

	aml_copy_rtnd(4, dst2, src_pitch, dst, dst_pitch, elem_number2,
		      sizeof(double));
	for (int l = 0; l < 8; l++)
		for (int k = 0; k < 4; k++)
			for (int j = 0; j < 6; j++)
				for (int i = 0; i < 10; i++)
					assert(ref_dst2[l][k][j][i] ==
					       dst2[l][k][j][i]);

}

void test_copy_t4d_c(void)
{
	size_t elem_number[4] = { 5, 3, 2, 4 };
	size_t elem_number2[4] = { 3, 2, 4, 5 };
	size_t c_src_pitch[5] = { 8, 8 * 10, 8 * 10 * 6, 8 * 10 * 6 * 4,
				  8 * 10 * 6 * 4 * 8 };
	size_t c_dst_pitch[5] = { 8, 8 * 3, 8 * 3 * 2, 8 * 3 * 2 * 4,
				  8 * 3 * 2 * 4 * 5 };

	double src[8][4][6][10];
	double dst[5][4][2][3];
	double dst2[8][4][6][10];

	double ref_dst[5][4][2][3];
	double ref_dst2[8][4][6][10];

	for (int l = 0; l < 8; l++)
		for (int k = 0; k < 4; k++)
			for (int j = 0; j < 6; j++)
				for (int i = 0; i < 10; i++) {
					src[l][k][j][i] =
					    (double)(i + j * 10 + k * 10 * 6 +
						     l * 10 * 6 * 4);
					ref_dst2[l][k][j][i] = 0.0;
					dst2[l][k][j][i] = 0.0;
				}
	for (int l = 0; l < 4; l++)
		for (int k = 0; k < 2; k++)
			for (int j = 0; j < 3; j++)
				for (int i = 0; i < 5; i++) {
					dst[i][l][k][j] = 0.0;
					ref_dst[i][l][k][j] = src[l][k][j][i];
					ref_dst2[l][k][j][i] = src[l][k][j][i];
				}

	aml_copy_tnd_c(4, dst, c_dst_pitch, src, c_src_pitch, elem_number,
		     sizeof(double));
	for (int l = 0; l < 4; l++)
		for (int k = 0; k < 2; k++)
			for (int j = 0; j < 3; j++)
				for (int i = 0; i < 5; i++)
					assert(ref_dst[i][l][k][j] ==
					       dst[i][l][k][j]);

	aml_copy_rtnd_c(4, dst2, c_src_pitch, dst, c_dst_pitch, elem_number2,
		      sizeof(double));
	for (int l = 0; l < 8; l++)
		for (int k = 0; k < 4; k++)
			for (int j = 0; j < 6; j++)
				for (int i = 0; i < 10; i++)
					assert(ref_dst2[l][k][j][i] ==
					       dst2[l][k][j][i]);

}

void test_copy_t4dstr(void)
{
	size_t elem_number[4] = { 5, 3, 2, 4 };
	size_t elem_number2[4] = { 3, 2, 4, 5 };
	size_t src_pitch[4] = { 10, 6, 4, 8 };
	size_t src_stride[4] = { 2, 2, 2, 2 };
	size_t dst_pitch[4] = { 3, 2, 4, 5 };
	size_t dst_stride[4] = { 1, 1, 1, 1 };

	double src[8][4][6][10];
	double dst[5][4][2][3];
	double dst2[8][4][6][10];

	double ref_dst[5][4][2][3];
	double ref_dst2[8][4][6][10];

	for (int l = 0; l < 8; l++)
		for (int k = 0; k < 4; k++)
			for (int j = 0; j < 6; j++)
				for (int i = 0; i < 10; i++) {
					src[l][k][j][i] =
					    (double)(i + j * 10 + k * 10 * 6 +
						     l * 10 * 6 * 4);
					ref_dst2[l][k][j][i] = 0.0;
					dst2[l][k][j][i] = 0.0;
				}
	for (int l = 0; l < 4; l++)
		for (int k = 0; k < 2; k++)
			for (int j = 0; j < 3; j++)
				for (int i = 0; i < 5; i++) {
					dst[i][l][k][j] = 0.0;
					ref_dst[i][l][k][j] =
					    src[2 * l][2 * k][2 * j][2 * i];
					ref_dst2[2 * l][2 * k][2 * j][2 * i] =
					    src[2 * l][2 * k][2 * j][2 * i];
				}

	aml_copy_tndstr(4, dst, dst_pitch, dst_stride, src, src_pitch,
			src_stride, elem_number, sizeof(double));
	for (int l = 0; l < 4; l++)
		for (int k = 0; k < 2; k++)
			for (int j = 0; j < 3; j++)
				for (int i = 0; i < 5; i++)
					assert(ref_dst[i][l][k][j] ==
					       dst[i][l][k][j]);

	aml_copy_rtndstr(4, dst2, src_pitch, src_stride, dst, dst_pitch,
			 dst_stride, elem_number2, sizeof(double));
	for (int l = 0; l < 8; l++)
		for (int k = 0; k < 4; k++)
			for (int j = 0; j < 6; j++)
				for (int i = 0; i < 10; i++)
					assert(ref_dst2[l][k][j][i] ==
					       dst2[l][k][j][i]);

}

void test_copy_t4dstr_c(void)
{
	size_t elem_number[4] = { 5, 3, 2, 4 };
	size_t elem_number2[4] = { 3, 2, 4, 5 };
	size_t c_src_pitch[5] = { 8, 8 * 10, 8 * 10 * 6, 8 * 10 * 6 * 4,
				  8 * 10 * 6 * 4 * 8 };
	size_t src_stride[4] = { 2, 2, 2, 2 };
	size_t c_dst_pitch[5] = { 8, 8 * 3, 8 * 3 * 2, 8 * 3 * 2 * 4,
				  8 * 3 * 2 * 4 * 5 };
	size_t dst_stride[4] = { 1, 1, 1, 1 };

	double src[8][4][6][10];
	double dst[5][4][2][3];
	double dst2[8][4][6][10];

	double ref_dst[5][4][2][3];
	double ref_dst2[8][4][6][10];

	for (int l = 0; l < 8; l++)
		for (int k = 0; k < 4; k++)
			for (int j = 0; j < 6; j++)
				for (int i = 0; i < 10; i++) {
					src[l][k][j][i] =
					    (double)(i + j * 10 + k * 10 * 6 +
						     l * 10 * 6 * 4);
					ref_dst2[l][k][j][i] = 0.0;
					dst2[l][k][j][i] = 0.0;
				}
	for (int l = 0; l < 4; l++)
		for (int k = 0; k < 2; k++)
			for (int j = 0; j < 3; j++)
				for (int i = 0; i < 5; i++) {
					dst[i][l][k][j] = 0.0;
					ref_dst[i][l][k][j] =
					    src[2 * l][2 * k][2 * j][2 * i];
					ref_dst2[2 * l][2 * k][2 * j][2 * i] =
					    src[2 * l][2 * k][2 * j][2 * i];
				}

	aml_copy_tndstr_c(4, dst, c_dst_pitch, dst_stride, src, c_src_pitch,
			  src_stride, elem_number, sizeof(double));
	for (int l = 0; l < 4; l++)
		for (int k = 0; k < 2; k++)
			for (int j = 0; j < 3; j++)
				for (int i = 0; i < 5; i++)
					assert(ref_dst[i][l][k][j] ==
					       dst[i][l][k][j]);

	aml_copy_rtndstr_c(4, dst2, c_src_pitch, src_stride, dst, c_dst_pitch,
			   dst_stride, elem_number2, sizeof(double));
	for (int l = 0; l < 8; l++)
		for (int k = 0; k < 4; k++)
			for (int j = 0; j < 6; j++)
				for (int i = 0; i < 10; i++)
					assert(ref_dst2[l][k][j][i] ==
					       dst2[l][k][j][i]);

}

void test_copy_sh4d(void)
{
	size_t elem_number[4] = { 5, 3, 2, 4 };
	size_t elem_number2[4] = { 2, 3, 4, 5 };
	size_t target_dims[4] = { 3, 1, 0, 2 };
	size_t target_dims2[4] = { 2, 1, 3, 0 };
	size_t src_pitch[4] = { 10, 6, 4, 8 };
	size_t dst_pitch[4] = { 2, 3, 4, 5 };

	double src[8][4][6][10];
	double dst[5][4][3][2];
	double dst2[8][4][6][10];

	double ref_dst[5][4][3][2];
	double ref_dst2[8][4][6][10];

	for (int l = 0; l < 8; l++)
		for (int k = 0; k < 4; k++)
			for (int j = 0; j < 6; j++)
				for (int i = 0; i < 10; i++) {
					src[l][k][j][i] =
					    (double)(i + j * 10 + k * 10 * 6 +
						     l * 10 * 6 * 4);
					ref_dst2[l][k][j][i] = 0.0;
					dst2[l][k][j][i] = 0.0;
				}

	for (int l = 0; l < 4; l++)
		for (int k = 0; k < 2; k++)
			for (int j = 0; j < 3; j++)
				for (int i = 0; i < 5; i++) {
					dst[i][l][j][k] = 0.0;
					ref_dst[i][l][j][k] = src[l][k][j][i];
					ref_dst2[l][k][j][i] = src[l][k][j][i];
				}

	aml_copy_shnd(4, target_dims, dst, dst_pitch, src, src_pitch,
		      elem_number, sizeof(double));
	for (int l = 0; l < 4; l++)
		for (int k = 0; k < 2; k++)
			for (int j = 0; j < 3; j++)
				for (int i = 0; i < 5; i++)
					assert(ref_dst[i][l][j][k] ==
					       dst[i][l][j][k]);

	aml_copy_shnd(4, target_dims2, dst2, src_pitch, dst, dst_pitch,
		      elem_number2, sizeof(double));
	for (int l = 0; l < 8; l++)
		for (int k = 0; k < 4; k++)
			for (int j = 0; j < 6; j++)
				for (int i = 0; i < 10; i++)
					assert(ref_dst2[l][k][j][i] ==
					       dst2[l][k][j][i]);

}

void test_copy_sh4d_c(void)
{
	size_t elem_number[4] = { 5, 3, 2, 4 };
	size_t elem_number2[4] = { 2, 3, 4, 5 };
	size_t target_dims[4] = { 3, 1, 0, 2 };
	size_t target_dims2[4] = { 2, 1, 3, 0 };
	size_t c_src_pitch[5] = { 8, 8 * 10, 8 * 10 * 6, 8 * 10 * 6 * 4,
				  8 * 10 * 6 * 4 * 8 };
	size_t c_dst_pitch[5] = { 8, 8 * 2, 8 * 2 * 3, 8 * 2 * 3 * 4,
				  8 * 2 * 3 * 4 * 5 };

	double src[8][4][6][10];
	double dst[5][4][3][2];
	double dst2[8][4][6][10];

	double ref_dst[5][4][3][2];
	double ref_dst2[8][4][6][10];

	for (int l = 0; l < 8; l++)
		for (int k = 0; k < 4; k++)
			for (int j = 0; j < 6; j++)
				for (int i = 0; i < 10; i++) {
					src[l][k][j][i] =
					    (double)(i + j * 10 + k * 10 * 6 +
						     l * 10 * 6 * 4);
					ref_dst2[l][k][j][i] = 0.0;
					dst2[l][k][j][i] = 0.0;
				}

	for (int l = 0; l < 4; l++)
		for (int k = 0; k < 2; k++)
			for (int j = 0; j < 3; j++)
				for (int i = 0; i < 5; i++) {
					dst[i][l][j][k] = 0.0;
					ref_dst[i][l][j][k] = src[l][k][j][i];
					ref_dst2[l][k][j][i] = src[l][k][j][i];
				}

	aml_copy_shnd_c(4, target_dims, dst, c_dst_pitch, src, c_src_pitch,
			elem_number, sizeof(double));
	for (int l = 0; l < 4; l++)
		for (int k = 0; k < 2; k++)
			for (int j = 0; j < 3; j++)
				for (int i = 0; i < 5; i++)
					assert(ref_dst[i][l][j][k] ==
					       dst[i][l][j][k]);

	aml_copy_shnd_c(4, target_dims2, dst2, c_src_pitch, dst, c_dst_pitch,
			elem_number2, sizeof(double));
	for (int l = 0; l < 8; l++)
		for (int k = 0; k < 4; k++)
			for (int j = 0; j < 6; j++)
				for (int i = 0; i < 10; i++)
					assert(ref_dst2[l][k][j][i] ==
					       dst2[l][k][j][i]);

}

void test_copy_sh4dstr(void)
{
	size_t elem_number[4] = { 5, 3, 2, 4 };
	size_t elem_number2[4] = { 2, 3, 4, 5 };
	size_t target_dims[4] = { 3, 1, 0, 2 };
	size_t target_dims2[4] = { 2, 1, 3, 0 };
	size_t src_pitch[4] = { 10, 6, 4, 8 };
	size_t src_stride[4] = { 2, 2, 2, 2 };
	size_t dst_pitch[4] = { 2, 3, 4, 5 };
	size_t dst_stride[4] = { 1, 1, 1, 1 };

	double src[8][4][6][10];
	double dst[5][4][3][2];
	double dst2[8][4][6][10];

	double ref_dst[5][4][3][2];
	double ref_dst2[8][4][6][10];

	for (int l = 0; l < 8; l++)
		for (int k = 0; k < 4; k++)
			for (int j = 0; j < 6; j++)
				for (int i = 0; i < 10; i++) {
					src[l][k][j][i] =
					    (double)(i + j * 10 + k * 10 * 6 +
						     l * 10 * 6 * 4);
					ref_dst2[l][k][j][i] = 0.0;
					dst2[l][k][j][i] = 0.0;
				}

	for (int l = 0; l < 4; l++)
		for (int k = 0; k < 2; k++)
			for (int j = 0; j < 3; j++)
				for (int i = 0; i < 5; i++) {
					dst[i][l][j][k] = 0.0;
					ref_dst[i][l][j][k] =
					    src[2 * l][2 * k][2 * j][2 * i];
					ref_dst2[2 * l][2 * k][2 * j][2 * i] =
					    src[2 * l][2 * k][2 * j][2 * i];
				}

	aml_copy_shndstr(4, target_dims, dst, dst_pitch, dst_stride, src,
			 src_pitch, src_stride, elem_number, sizeof(double));
	for (int l = 0; l < 4; l++)
		for (int k = 0; k < 2; k++)
			for (int j = 0; j < 3; j++)
				for (int i = 0; i < 5; i++)
					assert(ref_dst[i][l][j][k] ==
					       dst[i][l][j][k]);

	aml_copy_shndstr(4, target_dims2, dst2, src_pitch, src_stride, dst,
			 dst_pitch, dst_stride, elem_number2, sizeof(double));
	for (int l = 0; l < 8; l++)
		for (int k = 0; k < 4; k++)
			for (int j = 0; j < 6; j++)
				for (int i = 0; i < 10; i++)
					assert(ref_dst2[l][k][j][i] ==
					       dst2[l][k][j][i]);

}

void test_copy_sh4dstr_c(void)
{
	size_t elem_number[4] = { 5, 3, 2, 4 };
	size_t elem_number2[4] = { 2, 3, 4, 5 };
	size_t target_dims[4] = { 3, 1, 0, 2 };
	size_t target_dims2[4] = { 2, 1, 3, 0 };
	size_t c_src_pitch[5] = { 8, 8 * 10, 8 * 10 * 6, 8 * 10 * 6 * 4,
				  8 * 10 * 6 * 4 * 8 };
	size_t src_stride[4] = { 2, 2, 2, 2 };
	size_t c_dst_pitch[5] = { 8, 8 * 2, 8 * 2 * 3, 8 * 2 * 3 * 4,
				  8 * 2 * 3 * 4 * 5 };
	size_t dst_stride[4] = { 1, 1, 1, 1 };

	double src[8][4][6][10];
	double dst[5][4][3][2];
	double dst2[8][4][6][10];

	double ref_dst[5][4][3][2];
	double ref_dst2[8][4][6][10];

	for (int l = 0; l < 8; l++)
		for (int k = 0; k < 4; k++)
			for (int j = 0; j < 6; j++)
				for (int i = 0; i < 10; i++) {
					src[l][k][j][i] =
					    (double)(i + j * 10 + k * 10 * 6 +
						     l * 10 * 6 * 4);
					ref_dst2[l][k][j][i] = 0.0;
					dst2[l][k][j][i] = 0.0;
				}

	for (int l = 0; l < 4; l++)
		for (int k = 0; k < 2; k++)
			for (int j = 0; j < 3; j++)
				for (int i = 0; i < 5; i++) {
					dst[i][l][j][k] = 0.0;
					ref_dst[i][l][j][k] =
					    src[2 * l][2 * k][2 * j][2 * i];
					ref_dst2[2 * l][2 * k][2 * j][2 * i] =
					    src[2 * l][2 * k][2 * j][2 * i];
				}

	aml_copy_shndstr_c(4, target_dims, dst, c_dst_pitch, dst_stride, src,
			   c_src_pitch, src_stride, elem_number,
			   sizeof(double));
	for (int l = 0; l < 4; l++)
		for (int k = 0; k < 2; k++)
			for (int j = 0; j < 3; j++)
				for (int i = 0; i < 5; i++)
					assert(ref_dst[i][l][j][k] ==
					       dst[i][l][j][k]);

	aml_copy_shndstr_c(4, target_dims2, dst2, c_src_pitch, src_stride, dst,
			   c_dst_pitch, dst_stride, elem_number2,
			   sizeof(double));
	for (int l = 0; l < 8; l++)
		for (int k = 0; k < 4; k++)
			for (int j = 0; j < 6; j++)
				for (int i = 0; i < 10; i++)
					assert(ref_dst2[l][k][j][i] ==
					       dst2[l][k][j][i]);

}

void test_copy_layout(void)
{
	size_t elem_number[3] = { 5, 3, 2 };
	size_t c_src_pitch[4] = { 8, 8 * 10, 8 * 10 * 6, 8 * 10 * 6 * 4 };
	size_t src_stride[3] = { 1, 1, 1};
	size_t c_dst_pitch[4] = { 8, 8 * 5, 8 * 5 * 3, 8 * 5 * 3 * 2 };
	size_t dst_stride[3] = { 1, 1, 1};

	double src[4][6][10];
	double dst[2][3][5];
	double dst2[4][6][10];

	double ref_dst2[4][6][10];
	double ref_dst[2][3][5];

	struct aml_layout src_layout = {
		(void *)src,
		3,
		elem_number,
		c_src_pitch,
		src_stride
	};
	struct aml_layout dst_layout = {
		(void *)dst,
		3,
		elem_number,
		c_dst_pitch,
		dst_stride
	};
	struct aml_layout dst2_layout = {
		(void *)dst2,
		3,
		elem_number,
		c_src_pitch,
		src_stride
	};

	for (int k = 0; k < 4; k++)
		for (int j = 0; j < 6; j++)
			for (int i = 0; i < 10; i++) {
				src[k][j][i] =
				    (double)(i + j * 10 + k * 10 * 6);
				ref_dst2[k][j][i] = 0.0;
				dst2[k][j][i] = 0.0;
			}
	for (int k = 0; k < 2; k++)
		for (int j = 0; j < 3; j++)
			for (int i = 0; i < 5; i++) {
				dst[k][j][i] = 0.0;
				ref_dst[k][j][i] = src[k][j][i];
				ref_dst2[k][j][i] = src[k][j][i];
			}

	aml_copy_layout(&dst_layout, &src_layout);
	for (int k = 0; k < 2; k++)
		for (int j = 0; j < 3; j++)
			for (int i = 0; i < 5; i++)
				assert(ref_dst[k][j][i] == dst[k][j][i]);

	aml_copy_layout(&dst2_layout, &dst_layout);
	for (int k = 0; k < 4; k++)
		for (int j = 0; j < 6; j++)
			for (int i = 0; i < 10; i++)
				assert(ref_dst2[k][j][i] == dst2[k][j][i]);

}

void test_transpose_layout(void)
{
	size_t elem_number[4] = { 5, 3, 2, 4 };
	size_t elem_number2[4] = { 3, 2, 4, 5 };
	size_t c_src_pitch[5] = { 8, 8 * 10, 8 * 10 * 6, 8 * 10 * 6 * 4,
				  8 * 10 * 6 * 4 * 8 };
	size_t src_stride[4] = { 2, 2, 2, 2 };
	size_t c_dst_pitch[5] = { 8, 8 * 3, 8 * 3 * 2, 8 * 3 * 2 * 4,
				  8 * 3 * 2 * 4 * 5 };
	size_t dst_stride[4] = { 1, 1, 1, 1 };

	double src[8][4][6][10];
	double dst[5][4][2][3];
	double dst2[8][4][6][10];

	double ref_dst[5][4][2][3];
	double ref_dst2[8][4][6][10];

	struct aml_layout src_layout = {
		(void *)src,
		4,
		elem_number,
		c_src_pitch,
		src_stride
	};
	struct aml_layout dst_layout = {
		(void *)dst,
		4,
		elem_number2,
		c_dst_pitch,
		dst_stride
	};
	struct aml_layout dst2_layout = {
		(void *)dst2,
		4,
		elem_number,
		c_src_pitch,
		src_stride
	};

	for (int l = 0; l < 8; l++)
		for (int k = 0; k < 4; k++)
			for (int j = 0; j < 6; j++)
				for (int i = 0; i < 10; i++) {
					src[l][k][j][i] =
					    (double)(i + j * 10 + k * 10 * 6 +
						     l * 10 * 6 * 4);
					ref_dst2[l][k][j][i] = 0.0;
					dst2[l][k][j][i] = 0.0;
				}
	for (int l = 0; l < 4; l++)
		for (int k = 0; k < 2; k++)
			for (int j = 0; j < 3; j++)
				for (int i = 0; i < 5; i++) {
					dst[i][l][k][j] = 0.0;
					ref_dst[i][l][k][j] =
					    src[2 * l][2 * k][2 * j][2 * i];
					ref_dst2[2 * l][2 * k][2 * j][2 * i] =
					    src[2 * l][2 * k][2 * j][2 * i];
				}

	aml_transpose_layout(&dst_layout, &src_layout);
	for (int l = 0; l < 4; l++)
		for (int k = 0; k < 2; k++)
			for (int j = 0; j < 3; j++)
				for (int i = 0; i < 5; i++)
					assert(ref_dst[i][l][k][j] ==
					       dst[i][l][k][j]);

	aml_reverse_transpose_layout(&dst2_layout, &dst_layout);
	for (int l = 0; l < 8; l++)
		for (int k = 0; k < 4; k++)
			for (int j = 0; j < 6; j++)
				for (int i = 0; i < 10; i++)
					assert(ref_dst2[l][k][j][i] ==
					       dst2[l][k][j][i]);

}

int main(int argc, char *argv[])
{
	test_copy_2d();
	test_copy_t2d();
	test_copy_3d();
	test_copy_3d_c();
	test_copy_3dstr();
	test_copy_3dstr_c();
	test_copy_t3d();
	test_copy_rt3d();
	test_copy_t4d();
	test_copy_t4d_c();
	test_copy_t4dstr();
	test_copy_t4dstr_c();
	test_copy_sh4d();
	test_copy_sh4d_c();
	test_copy_sh4dstr();
	test_copy_sh4dstr_c();
	test_copy_layout();
	test_transpose_layout();
	return 0;
}
