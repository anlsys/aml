#include <aml.h>
#include <assert.h>
#include <errno.h>
#include <mkl.h>
#include <omp.h>
#include <pthread.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>

AML_TILING_2D_CONTIG_ROWMAJOR_DECL(tiling_inner);
AML_TILING_2D_CONTIG_ROWMAJOR_DECL(tiling_prefetch);
AML_AREA_LINUX_DECL(srcarea);
AML_AREA_LINUX_DECL(carea);
AML_AREA_LINUX_DECL(abarea);
AML_SCRATCH_PAR_DECL(sa);
AML_SCRATCH_PAR_DECL(sb);
AML_SCRATCH_PAR_DECL(sc);

size_t memsize, tilesize, blocksize, N, B, nB, T, nT;
const char *abmask,  *cmask;
double *a, *b, *c;
struct timespec start, stop;

typedef struct matrix_split_s {
	size_t size;
	size_t block_size;
        size_t block_number;
	size_t tile_size;
	size_t tile_number;
        const char *cmask;
        const char *abmask;
} matrix_split_t;


void gemm_256_256_256(const double * __restrict__ a, const double * __restrict__ b, double * __restrict__ c);
void gemm_320_320_320(const double * __restrict__ a, const double * __restrict__ b, double * __restrict__ c);
void gemm_384_384_384(const double * __restrict__ a, const double * __restrict__ b, double * __restrict__ c);
void gemm_448_448_448(const double * __restrict__ a, const double * __restrict__ b, double * __restrict__ c);
void gemm_512_512_512(const double * __restrict__ a, const double * __restrict__ b, double * __restrict__ c);

void (*gemm_ptr)(const double * __restrict__ a, const double * __restrict__ b, double * __restrict__ c) = (void (*)(const double *, const double *, double *))0xdeadbeef;


static const int size_number = 277;
matrix_split_t matrix_splits[277] = {
	{8192, 8192, 1, 256, 32, "1", "0"},
	{8320, 8320, 1, 320, 26, "1", "0"},
	{8448, 8448, 1, 256, 33, "1", "0"},
	{8512, 8512, 1, 448, 19, "1", "0"},
	{8640, 8640, 1, 320, 27, "1", "0"},
	{8704, 8704, 1, 256, 34, "1", "0"},
	{8832, 8832, 1, 384, 23, "1", "0"},
	{8960, 8960, 1, 256, 35, "1", "0"},
	{9216, 9216, 1, 384, 24, "1", "0"},
	{9280, 9280, 1, 320, 29, "1", "1"},
	{9408, 9408, 1, 448, 21, "1", "0"},
	{9472, 9472, 1, 256, 37, "1", "0"},
	{9600, 9600, 1, 384, 25, "1", "0"},
	{9728, 9728, 1, 256, 38, "1", "0"},
	{9856, 9856, 1, 448, 22, "1", "0"},
	{9920, 9920, 1, 320, 31, "1", "1"},
	{9984, 9984, 1, 384, 26, "1", "0"},
	{10240, 10240, 1, 256, 40, "1", "0"},
	{10304, 10304, 1, 448, 23, "1", "0"},
	{10368, 10368, 1, 384, 27, "1", "0"},
	{10496, 10496, 1, 256, 41, "1", "0"},
	{10560, 10560, 1, 320, 33, "1", "0"},
	{10752, 10752, 1, 384, 28, "1", "1"},
	{10880, 10880, 1, 320, 34, "1", "0"},
	{11008, 11008, 1, 256, 43, "1", "0"},
	{11136, 11136, 1, 384, 29, "1", "0"},
	{11200, 11200, 1, 448, 25, "1", "0"},
	{11264, 11264, 1, 256, 44, "1", "0"},
	{11520, 11520, 1, 384, 30, "1", "0"},
	{11648, 11648, 1, 448, 26, "1", "0"},
	{11776, 11776, 1, 256, 46, "1", "0"},
	{11840, 11840, 1, 320, 37, "1", "0"},
	{11904, 11904, 1, 384, 31, "1", "0"},
	{12032, 12032, 1, 256, 47, "1", "0"},
	{12096, 12096, 1, 448, 27, "1", "0"},
	{12160, 12160, 1, 320, 38, "1", "0"},
	{12288, 12288, 1, 384, 32, "1", "0"},
	{12480, 12480, 1, 320, 39, "1", "0"},
	{12544, 12544, 1, 256, 49, "1", "0"},
	{12672, 12672, 1, 384, 33, "1", "0"},
	{12800, 12800, 1, 256, 50, "1", "0"},
	{12992, 12992, 1, 448, 29, "1", "0"},
	{13056, 13056, 1, 384, 34, "1", "0"},
	{13120, 13120, 1, 320, 41, "0", "1"},
	{13312, 13312, 1, 256, 52, "1", "0"},
	{13440, 13440, 1, 384, 35, "1", "0"},
	{13568, 13568, 1, 256, 53, "1", "0"},
	{13760, 13760, 1, 320, 43, "0", "1"},
	{13824, 13824, 1, 256, 54, "1", "0"},
	{13888, 13888, 1, 448, 31, "1", "0"},
	{14080, 14080, 1, 256, 55, "1", "1"},
	{14208, 14208, 1, 384, 37, "1", "0"},
	{14336, 14336, 1, 256, 56, "1", "0"},
	{14400, 14400, 1, 320, 45, "0", "1"},
	{14592, 14592, 1, 256, 57, "1", "0"},
	{14720, 14720, 1, 320, 46, "1", "0"},
	{14784, 14784, 1, 448, 33, "1", "0"},
	{14848, 14848, 1, 256, 58, "1", "0"},
	{14976, 14976, 1, 384, 39, "1", "0"},
	{15040, 15040, 1, 320, 47, "1", "1"},
	{15104, 15104, 1, 256, 59, "1", "0"},
	{15232, 15232, 1, 448, 34, "1", "1"},
	{15360, 15360, 1, 256, 60, "1", "0"},
	{15616, 15616, 1, 256, 61, "1", "0"},
	{15680, 15680, 1, 448, 35, "1", "0"},
	{15744, 15744, 1, 384, 41, "1", "0"},
	{15872, 15872, 1, 256, 62, "1", "0"},
	{16000, 16000, 1, 320, 50, "0", "1"},
	{16128, 16128, 1, 256, 63, "1", "0"},
	{16320, 16320, 1, 320, 51, "1", "1"},
	{16384, 16384, 1, 256, 64, "1", "0"},
	{16640, 8320, 2, 320, 26, "1", "0"},
	{16896, 8448, 2, 256, 33, "1", "0"},
	{17024, 8512, 2, 448, 19, "1", "0"},
	{17280, 8640, 2, 320, 27, "1", "0"},
	{17408, 8704, 2, 256, 34, "1", "0"},
	{17664, 8832, 2, 384, 23, "1", "0"},
	{17920, 8960, 2, 256, 35, "1", "0"},
	{18432, 9216, 2, 384, 24, "1", "0"},
	{18560, 9280, 2, 320, 29, "1", "1"},
	{18816, 9408, 2, 448, 21, "1", "0"},
	{18944, 9472, 2, 256, 37, "1", "0"},
	{19200, 9600, 2, 384, 25, "1", "0"},
	{19456, 9728, 2, 256, 38, "1", "0"},
	{19712, 9856, 2, 448, 22, "1", "0"},
	{19840, 9920, 2, 320, 31, "1", "1"},
	{19968, 9984, 2, 384, 26, "1", "0"},
	{20480, 10240, 2, 256, 40, "1", "0"},
	{20608, 10304, 2, 448, 23, "1", "0"},
	{20736, 10368, 2, 384, 27, "1", "0"},
	{20992, 10496, 2, 256, 41, "1", "0"},
	{21120, 10560, 2, 320, 33, "1", "0"},
	{21504, 10752, 2, 384, 28, "1", "1"},
	{21760, 10880, 2, 320, 34, "1", "0"},
	{22016, 11008, 2, 256, 43, "1", "0"},
	{22272, 11136, 2, 384, 29, "1", "0"},
	{22400, 11200, 2, 448, 25, "1", "0"},
	{22528, 11264, 2, 256, 44, "1", "0"},
	{23040, 11520, 2, 384, 30, "1", "0"},
	{23296, 11648, 2, 448, 26, "1", "0"},
	{23552, 11776, 2, 256, 46, "1", "0"},
	{23680, 11840, 2, 320, 37, "1", "0"},
	{23808, 11904, 2, 384, 31, "1", "0"},
	{24064, 12032, 2, 256, 47, "1", "0"},
	{24192, 12096, 2, 448, 27, "1", "0"},
	{24320, 12160, 2, 320, 38, "1", "0"},
	{24576, 12288, 2, 384, 32, "1", "0"},
	{24960, 12480, 2, 320, 39, "1", "0"},
	{25088, 12544, 2, 256, 49, "1", "0"},
	{25344, 12672, 2, 384, 33, "1", "0"},
	{25536, 8512, 3, 448, 19, "1", "0"},
	{25600, 12800, 2, 256, 50, "1", "0"},
	{25920, 8640, 3, 320, 27, "1", "0"},
	{25984, 12992, 2, 448, 29, "1", "0"},
	{26112, 13056, 2, 384, 34, "1", "0"},
	{26240, 13120, 2, 320, 41, "0", "1"},
	{26496, 8832, 3, 384, 23, "1", "0"},
	{26624, 13312, 2, 256, 52, "1", "0"},
	{26880, 13440, 2, 384, 35, "1", "0"},
	{27136, 13568, 2, 256, 53, "1", "0"},
	{27520, 13760, 2, 320, 43, "0", "1"},
	{27648, 13824, 2, 256, 54, "1", "0"},
	{27776, 13888, 2, 448, 31, "1", "0"},
	{27840, 9280, 3, 320, 29, "1", "1"},
	{28160, 14080, 2, 256, 55, "1", "1"},
	{28224, 9408, 3, 448, 21, "1", "0"},
	{28416, 14208, 2, 384, 37, "1", "0"},
	{28672, 14336, 2, 256, 56, "1", "0"},
	{28800, 14400, 2, 320, 45, "0", "1"},
	{29184, 14592, 2, 256, 57, "1", "0"},
	{29440, 14720, 2, 320, 46, "1", "0"},
	{29568, 14784, 2, 448, 33, "1", "0"},
	{29696, 14848, 2, 256, 58, "1", "0"},
	{29760, 9920, 3, 320, 31, "1", "1"},
	{29952, 14976, 2, 384, 39, "1", "0"},
	{30080, 15040, 2, 320, 47, "1", "1"},
	{30208, 15104, 2, 256, 59, "1", "0"},
	{30464, 15232, 2, 448, 34, "1", "1"},
	{30720, 15360, 2, 256, 60, "1", "0"},
	{30912, 10304, 3, 448, 23, "1", "0"},
	{31104, 10368, 3, 384, 27, "1", "0"},
	{31232, 15616, 2, 256, 61, "1", "0"},
	{31360, 15680, 2, 448, 35, "1", "0"},
	{31488, 15744, 2, 384, 41, "1", "0"},
	{31680, 10560, 3, 320, 33, "1", "0"},
	{31744, 15872, 2, 256, 62, "1", "0"},
	{32000, 16000, 2, 320, 50, "0", "1"},
	{32256, 16128, 2, 256, 63, "1", "0"},
	{32640, 16320, 2, 320, 51, "1", "1"},
	{32768, 16384, 2, 256, 64, "1", "0"},
	{33024, 11008, 3, 256, 43, "1", "0"},
	{33280, 8320, 4, 320, 26, "1", "0"},
	{33408, 11136, 3, 384, 29, "1", "0"},
	{33600, 11200, 3, 448, 25, "1", "0"},
	{33792, 11264, 3, 256, 44, "1", "0"},
	{34048, 8512, 4, 448, 19, "1", "0"},
	{34560, 11520, 3, 384, 30, "1", "0"},
	{34816, 8704, 4, 256, 34, "1", "0"},
	{34944, 11648, 3, 448, 26, "1", "0"},
	{35328, 11776, 3, 256, 46, "1", "0"},
	{35520, 11840, 3, 320, 37, "1", "0"},
	{35712, 11904, 3, 384, 31, "1", "0"},
	{35840, 8960, 4, 256, 35, "1", "0"},
	{36096, 12032, 3, 256, 47, "1", "0"},
	{36288, 12096, 3, 448, 27, "1", "0"},
	{36480, 12160, 3, 320, 38, "1", "0"},
	{36864, 12288, 3, 384, 32, "1", "0"},
	{37120, 9280, 4, 320, 29, "1", "1"},
	{37440, 12480, 3, 320, 39, "1", "0"},
	{37632, 12544, 3, 256, 49, "1", "0"},
	{37888, 9472, 4, 256, 37, "1", "0"},
	{38016, 12672, 3, 384, 33, "1", "0"},
	{38400, 12800, 3, 256, 50, "1", "0"},
	{38912, 9728, 4, 256, 38, "1", "0"},
	{38976, 12992, 3, 448, 29, "1", "0"},
	{39168, 13056, 3, 384, 34, "1", "0"},
	{39360, 13120, 3, 320, 41, "0", "1"},
	{39424, 9856, 4, 448, 22, "1", "0"},
	{39680, 9920, 4, 320, 31, "1", "1"},
	{39936, 13312, 3, 256, 52, "1", "0"},
	{40320, 13440, 3, 384, 35, "1", "0"},
	{40704, 13568, 3, 256, 53, "1", "0"},
	{40960, 10240, 4, 256, 40, "1", "0"},
	{41216, 10304, 4, 448, 23, "1", "0"},
	{41280, 13760, 3, 320, 43, "0", "1"},
	{41472, 13824, 3, 256, 54, "1", "0"},
	{41600, 8320, 5, 320, 26, "1", "0"},
	{41664, 13888, 3, 448, 31, "1", "0"},
	{41984, 10496, 4, 256, 41, "1", "0"},
	{42240, 14080, 3, 256, 55, "1", "1"},
	{42560, 8512, 5, 448, 19, "1", "0"},
	{42624, 14208, 3, 384, 37, "1", "0"},
	{43008, 14336, 3, 256, 56, "1", "0"},
	{43200, 14400, 3, 320, 45, "0", "1"},
	{43520, 10880, 4, 320, 34, "1", "0"},
	{43776, 14592, 3, 256, 57, "1", "0"},
	{44032, 11008, 4, 256, 43, "1", "0"},
	{44160, 14720, 3, 320, 46, "1", "0"},
	{44352, 14784, 3, 448, 33, "1", "0"},
	{44544, 14848, 3, 256, 58, "1", "0"},
	{44800, 11200, 4, 448, 25, "1", "0"},
	{44928, 14976, 3, 384, 39, "1", "0"},
	{45056, 11264, 4, 256, 44, "1", "0"},
	{45120, 15040, 3, 320, 47, "1", "1"},
	{45312, 15104, 3, 256, 59, "1", "0"},
	{45696, 15232, 3, 448, 34, "1", "1"},
	{46080, 15360, 3, 256, 60, "1", "0"},
	{46400, 9280, 5, 320, 29, "1", "1"},
	{46592, 11648, 4, 448, 26, "1", "0"},
	{46848, 15616, 3, 256, 61, "1", "0"},
	{47040, 15680, 3, 448, 35, "1", "0"},
	{47104, 11776, 4, 256, 46, "1", "0"},
	{47232, 15744, 3, 384, 41, "1", "0"},
	{47360, 11840, 4, 320, 37, "1", "0"},
	{47616, 15872, 3, 256, 62, "1", "0"},
	{48000, 16000, 3, 320, 50, "0", "1"},
	{48128, 12032, 4, 256, 47, "1", "0"},
	{48384, 16128, 3, 256, 63, "1", "0"},
	{48640, 12160, 4, 320, 38, "1", "0"},
	{48960, 16320, 3, 320, 51, "1", "1"},
	{49152, 16384, 3, 256, 64, "1", "0"},
	{49280, 9856, 5, 448, 22, "1", "0"},
	{49600, 9920, 5, 320, 31, "1", "1"},
	{49920, 12480, 4, 320, 39, "1", "0"},
	{50176, 12544, 4, 256, 49, "1", "0"},
	{50688, 12672, 4, 384, 33, "1", "0"},
	{51072, 8512, 6, 448, 19, "1", "0"},
	{51200, 12800, 4, 256, 50, "1", "0"},
	{51520, 10304, 5, 448, 23, "1", "0"},
	{51840, 10368, 5, 384, 27, "1", "0"},
	{51968, 12992, 4, 448, 29, "1", "0"},
	{52224, 13056, 4, 384, 34, "1", "0"},
	{52480, 13120, 4, 320, 41, "0", "1"},
	{52800, 10560, 5, 320, 33, "1", "0"},
	{52992, 8832, 6, 384, 23, "1", "0"},
	{53248, 13312, 4, 256, 52, "1", "0"},
	{53760, 13440, 4, 384, 35, "1", "0"},
	{54272, 13568, 4, 256, 53, "1", "0"},
	{54400, 10880, 5, 320, 34, "1", "0"},
	{55040, 13760, 4, 320, 43, "0", "1"},
	{55296, 13824, 4, 256, 54, "1", "0"},
	{55552, 13888, 4, 448, 31, "1", "0"},
	{55680, 11136, 5, 384, 29, "1", "0"},
	{56000, 11200, 5, 448, 25, "1", "0"},
	{56320, 14080, 4, 256, 55, "1", "1"},
	{56448, 9408, 6, 448, 21, "1", "0"},
	{56832, 14208, 4, 384, 37, "1", "0"},
	{57344, 14336, 4, 256, 56, "1", "0"},
	{57600, 14400, 4, 320, 45, "0", "1"},
	{58240, 11648, 5, 448, 26, "1", "0"},
	{58368, 14592, 4, 256, 57, "1", "0"},
	{58880, 14720, 4, 320, 46, "1", "0"},
	{59136, 14784, 4, 448, 33, "1", "0"},
	{59200, 11840, 5, 320, 37, "1", "0"},
	{59392, 14848, 4, 256, 58, "1", "0"},
	{59520, 11904, 5, 384, 31, "1", "0"},
	{59584, 8512, 7, 448, 19, "1", "0"},
	{59904, 14976, 4, 384, 39, "1", "0"},
	{60160, 15040, 4, 320, 47, "1", "1"},
	{60416, 15104, 4, 256, 59, "1", "0"},
	{60480, 12096, 5, 448, 27, "1", "0"},
	{60800, 12160, 5, 320, 38, "1", "0"},
	{60928, 15232, 4, 448, 34, "1", "1"},
	{61440, 15360, 4, 256, 60, "1", "0"},
	{61824, 10304, 6, 448, 23, "1", "0"},
	{62208, 10368, 6, 384, 27, "1", "0"},
	{62400, 12480, 5, 320, 39, "1", "0"},
	{62464, 15616, 4, 256, 61, "1", "0"},
	{62720, 15680, 4, 448, 35, "1", "0"},
	{62976, 15744, 4, 384, 41, "1", "0"},
	{63360, 12672, 5, 384, 33, "1", "0"},
	{63488, 15872, 4, 256, 62, "1", "0"},
	{64000, 16000, 4, 320, 50, "0", "1"},
	{64512, 16128, 4, 256, 63, "1", "0"},
	{64960, 12992, 5, 448, 29, "1", "0"},
	{65280, 16320, 4, 320, 51, "1", "1"},
	{65536, 16384, 4, 256, 64, "1", "0"}
};

void do_innerwork(const double *ap, const double *bp, const double *cp)
{
	int lda = (int)T, ldb, ldc;
	ldb = lda;
	ldc = lda;
	size_t ndims[2];
	aml_tiling_ndims(&tiling_inner, &ndims[0], &ndims[1]);

        #pragma omp parallel
	{
        #pragma omp master
	{
		for(int j = 0; j < ndims[1]; j++) {
			for(int i = 0; i < ndims[0]; i++) {
				for(int k = 0; k < ndims[1]; k++) {
					size_t aoff, boff, coff;
					double *aip, *bip, *cip;
					aoff = i*ndims[1] + k;
					boff = k*ndims[1] + j;
					coff = i*ndims[1] + j;
					aip = aml_tiling_tilestart(&tiling_inner, ap, aoff);
					bip = aml_tiling_tilestart(&tiling_inner, bp, boff);
					cip = aml_tiling_tilestart(&tiling_inner, cp, coff);
					#pragma omp task depend(in: aip[0:T*T], bip[0:T*T]) depend (inout: cip[0:T*T])
					{ (*gemm_ptr)(aip, bip, cip); }
					//{ cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ldc, lda, ldb, 1.0, aip, lda, bip, ldb, 1.0, cip, ldc); }
				}
			}
		}
	}
	}
}

int nextai(int i, int j, int k)
{
	if(k < nB -1)
	{
		return i*nB + k + 1; // a[i,k+1]
	}
	else
	{
		if(j < nB -1)
			return i*nB; // a[i, 0]
		else
			if(i < nB -1)
				return (i+1)*nB; // a[i+1, 0]
			else
				return -1;
	}
}

int nextbi(int i, int j, int k)
{
	if(k < nB -1)
	{
		return (k+1)*nB + j; // b[k+1, j]
	}
	else
	{
		if(j < nB -1)
			return j+1; // b[0, j+1]
		else
			if(i < nB -1)
				return 0; // b[0, 0]
			else
				return -1;
	}
}

int nextci(int i, int j)
{
	if(j < nB -1)
		return  i*nB + j + 1;// c[i, j+1]
	else
		if(i < nB -1)
			return (i+1)*nB; // c[i+1, 0]
		else
			return -1;
}

void do_work()
{
	int ai, bi, ci; // indexes in fast
	int oldai, oldbi, oldci; // previous scratch index
	int aoff, boff, coff, oldcoff; // offsets in slow
	void *abaseptr, *bbaseptr, *cbaseptr;
	struct aml_scratch_request *ar, *br, *cr, *cpush;
	abaseptr = aml_scratch_baseptr(&sa);
	bbaseptr = aml_scratch_baseptr(&sb);
	cbaseptr = aml_scratch_baseptr(&sc);
	aoff = 0; boff = 0; coff = 0; oldci = -1;
	aml_scratch_async_pull(&sa, &ar, abaseptr, &ai, a, aoff);
	aml_scratch_async_pull(&sb, &br, bbaseptr, &bi, b, boff);
	aml_scratch_async_pull(&sc, &cr, cbaseptr, &ci, c, coff);
        aml_scratch_wait(&sc, cr);
        aml_scratch_wait(&sa, ar);
        aml_scratch_wait(&sb, br);
	for(int i = 0; i < nB; i++)
	{
		for(int j = 0; j < nB; j++)
		{
			double *cp;
			coff = nextci(i, j);
			cp = aml_tiling_tilestart(&tiling_prefetch, cbaseptr, ci);
			if(i +j != 0) aml_scratch_async_push(&sc, &cpush, c, &oldcoff, cbaseptr, oldci);
			oldci = ci;
			if(coff != -1) aml_scratch_async_pull(&sc, &cr, cbaseptr, &ci, c, coff);
			for(int k = 0; k < nB; k++)
			{
				double *ap, *bp;
				ap = aml_tiling_tilestart(&tiling_prefetch, abaseptr, ai);
				bp = aml_tiling_tilestart(&tiling_prefetch, bbaseptr, bi);
				aoff = nextai(i, j, k);
				boff = nextbi(i, j, k);
				oldai = ai; oldbi = bi;
				if(aoff != -1) aml_scratch_async_pull(&sa, &ar, abaseptr, &ai, a, aoff);
				if(boff != -1) aml_scratch_async_pull(&sb, &br, bbaseptr, &bi, b, boff);
				do_innerwork(ap, bp, cp);
				if(aoff != -1) aml_scratch_wait(&sa, ar);
				if(boff != -1) aml_scratch_wait(&sb, br);
				aml_scratch_release(&sa, oldai);
				aml_scratch_release(&sb, oldbi);
			}
			if(coff != -1) aml_scratch_wait(&sc, cr);
			if(i + j != 0) aml_scratch_wait(&sc, cpush);
		}
	}
	aml_scratch_push(&sc, c, &oldcoff, cbaseptr, oldci);
}

void find_blocking_tiling()
{
	for (int i = 0; i < size_number; i++) {
		if (matrix_splits[i].size >= N) {
			N = matrix_splits[i].size;
			B = matrix_splits[i].block_size;
			T = matrix_splits[i].tile_size;
			nB = matrix_splits[i].block_number;
			nT = matrix_splits[i].tile_number;
                        abmask = matrix_splits[i].abmask;
                        cmask = matrix_splits[i].cmask;
			switch(T) {
			case 256:
				gemm_ptr = & gemm_256_256_256;
				break;
			case 320:
				gemm_ptr = & gemm_320_320_320;
				break;
			case 384:
				gemm_ptr = & gemm_384_384_384;
				break;
			case 448:
				gemm_ptr = & gemm_448_448_448;
				break;
			case 512:
				gemm_ptr = & gemm_512_512_512;
				break;
			}
			return;
		}
	}
	assert(0);
}

int main(int argc, char* argv[])
{
	AML_ARENA_JEMALLOC_DECL(arnsrc);
	AML_ARENA_JEMALLOC_DECL(arnab);
	AML_ARENA_JEMALLOC_DECL(arnc);
	AML_DMA_LINUX_SEQ_DECL(dma);
	struct bitmask *srcbm, *abbm, *cbm;
	aml_init(&argc, &argv);
	assert(argc == 2);
	N = atol(argv[1]);
	find_blocking_tiling();
	srcbm = numa_parse_nodestring_all("0");
	abbm = numa_parse_nodestring_all(abmask);
	cbm = numa_parse_nodestring_all(cmask);
	assert(N % T == 0);
	memsize = sizeof(double)*N*N;
	tilesize = sizeof(double)*T*T;
	blocksize = sizeof(double)*B*B;

	/* A inner tiling of each block, and a tiling of blocks  */
	assert(!aml_tiling_init(&tiling_inner, AML_TILING_TYPE_2D_CONTIG_ROWMAJOR,
				tilesize, blocksize, nT , nT));
	assert(!aml_tiling_init(&tiling_prefetch, AML_TILING_TYPE_2D_CONTIG_ROWMAJOR,
				blocksize, memsize, nB, nB));

	assert(!aml_arena_jemalloc_init(&arnsrc, AML_ARENA_JEMALLOC_TYPE_ALIGNED, (size_t)(64)));
	assert(!aml_area_linux_init(&srcarea,
				    AML_AREA_LINUX_MANAGER_TYPE_SINGLE,
				    AML_AREA_LINUX_MBIND_TYPE_REGULAR,
				    AML_AREA_LINUX_MMAP_TYPE_ANONYMOUS,
				    &arnsrc, MPOL_BIND, srcbm->maskp));
	assert(!aml_arena_jemalloc_init(&arnab, AML_ARENA_JEMALLOC_TYPE_ALIGNED, (size_t)(64)));
	assert(!aml_area_linux_init(&abarea,
				    AML_AREA_LINUX_MANAGER_TYPE_SINGLE,
				    AML_AREA_LINUX_MBIND_TYPE_REGULAR,
				    AML_AREA_LINUX_MMAP_TYPE_ANONYMOUS,
				    &arnab, MPOL_BIND, abbm->maskp));
	assert(!aml_arena_jemalloc_init(&arnc, AML_ARENA_JEMALLOC_TYPE_ALIGNED, (size_t)(64)));
	assert(!aml_area_linux_init(&carea,
				    AML_AREA_LINUX_MANAGER_TYPE_SINGLE,
				    AML_AREA_LINUX_MBIND_TYPE_REGULAR,
				    AML_AREA_LINUX_MMAP_TYPE_ANONYMOUS,
				    &arnc, MPOL_BIND, cbm->maskp));
	assert(!aml_dma_linux_seq_init(&dma, 4));
	assert(!aml_scratch_par_init(&sa, &abarea, &srcarea, &dma, &tiling_prefetch, (size_t)2, (size_t)2));
	assert(!aml_scratch_par_init(&sb, &abarea, &srcarea, &dma, &tiling_prefetch, (size_t)2, (size_t)2));
	assert(!aml_scratch_par_init(&sc, &carea, &srcarea, &dma, &tiling_prefetch, (size_t)3, (size_t)3));
	/* allocation */
	a = aml_area_malloc(&srcarea, memsize);
	b = aml_area_malloc(&srcarea, memsize);
	c = aml_area_malloc(&srcarea, memsize);
	assert(a != NULL && b != NULL && c != NULL);

	clock_gettime(CLOCK_REALTIME, &start);
	do_work();
	clock_gettime(CLOCK_REALTIME, &stop);
	long long int time = 0;
	time =  (stop.tv_nsec - start.tv_nsec) +
                1e9* (stop.tv_sec - start.tv_sec);
	double flops = (2.0*N*N*N)/(time/1e9);

	/* print the flops in GFLOPS */
	printf("dgemm-blocked: %llu %lld %lld %f\n", N, memsize, time,
	       flops/1e9);
	aml_scratch_par_destroy(&sa);
	aml_scratch_par_destroy(&sb);
	aml_scratch_par_destroy(&sc);
	aml_dma_linux_seq_destroy(&dma);
	aml_area_free(&srcarea, a);
	aml_area_free(&srcarea, b);
	aml_area_free(&srcarea, c);
	aml_area_linux_destroy(&srcarea);
	aml_area_linux_destroy(&abarea);
	aml_area_linux_destroy(&carea);
	aml_tiling_destroy(&tiling_inner, AML_TILING_TYPE_2D_CONTIG_ROWMAJOR);
	aml_tiling_destroy(&tiling_prefetch, AML_TILING_TYPE_2D_CONTIG_ROWMAJOR);
	aml_finalize();
	return 0;
}
