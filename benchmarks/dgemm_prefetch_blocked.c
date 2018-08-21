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
AML_AREA_LINUX_DECL(slow);
AML_AREA_LINUX_DECL(fast);
AML_SCRATCH_PAR_DECL(sa);
AML_SCRATCH_PAR_DECL(sb);

size_t memsize, tilesize, blocksize, N, B, nB, T, nT;
double *a, *b, *c;
struct timespec start, stop;

typedef struct matrix_split_s {
	size_t size;
	size_t block_size;
        size_t block_number;
	size_t tile_size;
	size_t tile_number;
} matrix_split_t;


static const int size_number = 277;
static matrix_split_t matrix_splits[size_number] = {
	{8192, 8192, 1, 512, 16},
	{8320, 8320, 1, 320, 26},
	{8448, 8448, 1, 384, 22},
	{8512, 8512, 1, 448, 19},
	{8640, 8640, 1, 320, 27},
	{8704, 8704, 1, 512, 17},
	{8832, 8832, 1, 384, 23},
	{8960, 8960, 1, 448, 20},
	{9216, 9216, 1, 512, 18},
	{9280, 9280, 1, 320, 29},
	{9408, 9408, 1, 448, 21},
	{9472, 9472, 1, 256, 37},
	{9600, 9600, 1, 384, 25},
	{9728, 9728, 1, 512, 19},
	{9856, 9856, 1, 448, 22},
	{9920, 9920, 1, 320, 31},
	{9984, 9984, 1, 384, 26},
	{10240, 10240, 1, 512, 20},
	{10304, 10304, 1, 448, 23},
	{10368, 10368, 1, 384, 27},
	{10496, 10496, 1, 256, 41},
	{10560, 10560, 1, 320, 33},
	{10752, 10752, 1, 512, 21},
	{10880, 10880, 1, 320, 34},
	{11008, 11008, 1, 256, 43},
	{11136, 11136, 1, 384, 29},
	{11200, 11200, 1, 448, 25},
	{11264, 11264, 1, 512, 22},
	{11520, 11520, 1, 384, 30},
	{11648, 11648, 1, 448, 26},
	{11776, 11776, 1, 512, 23},
	{11840, 11840, 1, 320, 37},
	{11904, 11904, 1, 384, 31},
	{12032, 12032, 1, 256, 47},
	{12096, 12096, 1, 448, 27},
	{12160, 12160, 1, 320, 38},
	{12288, 12288, 1, 512, 24},
	{12480, 12480, 1, 320, 39},
	{12544, 12544, 1, 448, 28},
	{12672, 12672, 1, 384, 33},
	{12800, 12800, 1, 512, 25},
	{12992, 12992, 1, 448, 29},
	{13056, 13056, 1, 384, 34},
	{13120, 13120, 1, 320, 41},
	{13312, 13312, 1, 512, 26},
	{13440, 13440, 1, 448, 30},
	{13568, 13568, 1, 256, 53},
	{13760, 13760, 1, 320, 43},
	{13824, 13824, 1, 512, 27},
	{13888, 13888, 1, 448, 31},
	{14080, 14080, 1, 320, 44},
	{14208, 14208, 1, 384, 37},
	{14336, 14336, 1, 512, 28},
	{14400, 14400, 1, 320, 45},
	{14592, 14592, 1, 384, 38},
	{14720, 14720, 1, 320, 46},
	{14784, 14784, 1, 448, 33},
	{14848, 14848, 1, 512, 29},
	{14976, 14976, 1, 384, 39},
	{15040, 15040, 1, 320, 47},
	{15104, 15104, 1, 256, 59},
	{15232, 15232, 1, 448, 34},
	{15360, 15360, 1, 512, 30},
	{15616, 15616, 1, 256, 61},
	{15680, 15680, 1, 448, 35},
	{15744, 15744, 1, 384, 41},
	{15872, 15872, 1, 512, 31},
	{16000, 16000, 1, 320, 50},
	{16128, 16128, 1, 448, 36},
	{16320, 16320, 1, 320, 51},
	{16384, 16384, 1, 512, 32},
	{16640, 8320, 2, 320, 26},
	{16896, 8448, 2, 384, 22},
	{17024, 8512, 2, 448, 19},
	{17280, 8640, 2, 320, 27},
	{17408, 8704, 2, 512, 17},
	{17664, 8832, 2, 384, 23},
	{17920, 8960, 2, 448, 20},
	{18432, 9216, 2, 512, 18},
	{18560, 9280, 2, 320, 29},
	{18816, 9408, 2, 448, 21},
	{18944, 9472, 2, 256, 37},
	{19200, 9600, 2, 384, 25},
	{19456, 9728, 2, 512, 19},
	{19712, 9856, 2, 448, 22},
	{19840, 9920, 2, 320, 31},
	{19968, 9984, 2, 384, 26},
	{20480, 10240, 2, 512, 20},
	{20608, 10304, 2, 448, 23},
	{20736, 10368, 2, 384, 27},
	{20992, 10496, 2, 256, 41},
	{21120, 10560, 2, 320, 33},
	{21504, 10752, 2, 512, 21},
	{21760, 10880, 2, 320, 34},
	{22016, 11008, 2, 256, 43},
	{22272, 11136, 2, 384, 29},
	{22400, 11200, 2, 448, 25},
	{22528, 11264, 2, 512, 22},
	{23040, 11520, 2, 384, 30},
	{23296, 11648, 2, 448, 26},
	{23552, 11776, 2, 512, 23},
	{23680, 11840, 2, 320, 37},
	{23808, 11904, 2, 384, 31},
	{24064, 12032, 2, 256, 47},
	{24192, 12096, 2, 448, 27},
	{24320, 12160, 2, 320, 38},
	{24576, 12288, 2, 512, 24},
	{24960, 12480, 2, 320, 39},
	{25088, 12544, 2, 448, 28},
	{25344, 12672, 2, 384, 33},
	{25536, 8512, 3, 448, 19},
	{25600, 12800, 2, 512, 25},
	{25920, 8640, 3, 320, 27},
	{25984, 12992, 2, 448, 29},
	{26112, 13056, 2, 384, 34},
	{26240, 13120, 2, 320, 41},
	{26496, 8832, 3, 384, 23},
	{26624, 13312, 2, 512, 26},
	{26880, 13440, 2, 448, 30},
	{27136, 13568, 2, 256, 53},
	{27520, 13760, 2, 320, 43},
	{27648, 13824, 2, 512, 27},
	{27776, 13888, 2, 448, 31},
	{27840, 9280, 3, 320, 29},
	{28160, 14080, 2, 320, 44},
	{28224, 9408, 3, 448, 21},
	{28416, 14208, 2, 384, 37},
	{28672, 14336, 2, 512, 28},
	{28800, 14400, 2, 320, 45},
	{29184, 14592, 2, 384, 38},
	{29440, 14720, 2, 320, 46},
	{29568, 14784, 2, 448, 33},
	{29696, 14848, 2, 512, 29},
	{29760, 9920, 3, 320, 31},
	{29952, 14976, 2, 384, 39},
	{30080, 15040, 2, 320, 47},
	{30208, 15104, 2, 256, 59},
	{30464, 15232, 2, 448, 34},
	{30720, 15360, 2, 512, 30},
	{30912, 10304, 3, 448, 23},
	{31104, 10368, 3, 384, 27},
	{31232, 15616, 2, 256, 61},
	{31360, 15680, 2, 448, 35},
	{31488, 15744, 2, 384, 41},
	{31680, 10560, 3, 320, 33},
	{31744, 15872, 2, 512, 31},
	{32000, 16000, 2, 320, 50},
	{32256, 16128, 2, 448, 36},
	{32640, 16320, 2, 320, 51},
	{32768, 16384, 2, 512, 32},
	{33024, 11008, 3, 256, 43},
	{33280, 8320, 4, 320, 26},
	{33408, 11136, 3, 384, 29},
	{33600, 11200, 3, 448, 25},
	{33792, 11264, 3, 512, 22},
	{34048, 8512, 4, 448, 19},
	{34560, 11520, 3, 384, 30},
	{34816, 8704, 4, 512, 17},
	{34944, 11648, 3, 448, 26},
	{35328, 11776, 3, 512, 23},
	{35520, 11840, 3, 320, 37},
	{35712, 11904, 3, 384, 31},
	{35840, 8960, 4, 448, 20},
	{36096, 12032, 3, 256, 47},
	{36288, 12096, 3, 448, 27},
	{36480, 12160, 3, 320, 38},
	{36864, 12288, 3, 512, 24},
	{37120, 9280, 4, 320, 29},
	{37440, 12480, 3, 320, 39},
	{37632, 12544, 3, 448, 28},
	{37888, 9472, 4, 256, 37},
	{38016, 12672, 3, 384, 33},
	{38400, 12800, 3, 512, 25},
	{38912, 9728, 4, 512, 19},
	{38976, 12992, 3, 448, 29},
	{39168, 13056, 3, 384, 34},
	{39360, 13120, 3, 320, 41},
	{39424, 9856, 4, 448, 22},
	{39680, 9920, 4, 320, 31},
	{39936, 13312, 3, 512, 26},
	{40320, 13440, 3, 448, 30},
	{40704, 13568, 3, 256, 53},
	{40960, 10240, 4, 512, 20},
	{41216, 10304, 4, 448, 23},
	{41280, 13760, 3, 320, 43},
	{41472, 13824, 3, 512, 27},
	{41600, 8320, 5, 320, 26},
	{41664, 13888, 3, 448, 31},
	{41984, 10496, 4, 256, 41},
	{42240, 14080, 3, 320, 44},
	{42560, 8512, 5, 448, 19},
	{42624, 14208, 3, 384, 37},
	{43008, 14336, 3, 512, 28},
	{43200, 14400, 3, 320, 45},
	{43520, 10880, 4, 320, 34},
	{43776, 14592, 3, 384, 38},
	{44032, 11008, 4, 256, 43},
	{44160, 14720, 3, 320, 46},
	{44352, 14784, 3, 448, 33},
	{44544, 14848, 3, 512, 29},
	{44800, 11200, 4, 448, 25},
	{44928, 14976, 3, 384, 39},
	{45056, 11264, 4, 512, 22},
	{45120, 15040, 3, 320, 47},
	{45312, 15104, 3, 256, 59},
	{45696, 15232, 3, 448, 34},
	{46080, 15360, 3, 512, 30},
	{46400, 9280, 5, 320, 29},
	{46592, 11648, 4, 448, 26},
	{46848, 15616, 3, 256, 61},
	{47040, 15680, 3, 448, 35},
	{47104, 11776, 4, 512, 23},
	{47232, 15744, 3, 384, 41},
	{47360, 11840, 4, 320, 37},
	{47616, 15872, 3, 512, 31},
	{48000, 16000, 3, 320, 50},
	{48128, 12032, 4, 256, 47},
	{48384, 16128, 3, 448, 36},
	{48640, 12160, 4, 320, 38},
	{48960, 16320, 3, 320, 51},
	{49152, 16384, 3, 512, 32},
	{49280, 9856, 5, 448, 22},
	{49600, 9920, 5, 320, 31},
	{49920, 12480, 4, 320, 39},
	{50176, 12544, 4, 448, 28},
	{50688, 12672, 4, 384, 33},
	{51072, 8512, 6, 448, 19},
	{51200, 12800, 4, 512, 25},
	{51520, 10304, 5, 448, 23},
	{51840, 10368, 5, 384, 27},
	{51968, 12992, 4, 448, 29},
	{52224, 13056, 4, 384, 34},
	{52480, 13120, 4, 320, 41},
	{52800, 10560, 5, 320, 33},
	{52992, 8832, 6, 384, 23},
	{53248, 13312, 4, 512, 26},
	{53760, 13440, 4, 448, 30},
	{54272, 13568, 4, 256, 53},
	{54400, 10880, 5, 320, 34},
	{55040, 13760, 4, 320, 43},
	{55296, 13824, 4, 512, 27},
	{55552, 13888, 4, 448, 31},
	{55680, 11136, 5, 384, 29},
	{56000, 11200, 5, 448, 25},
	{56320, 14080, 4, 320, 44},
	{56448, 9408, 6, 448, 21},
	{56832, 14208, 4, 384, 37},
	{57344, 14336, 4, 512, 28},
	{57600, 14400, 4, 320, 45},
	{58240, 11648, 5, 448, 26},
	{58368, 14592, 4, 384, 38},
	{58880, 14720, 4, 320, 46},
	{59136, 14784, 4, 448, 33},
	{59200, 11840, 5, 320, 37},
	{59392, 14848, 4, 512, 29},
	{59520, 11904, 5, 384, 31},
	{59584, 8512, 7, 448, 19},
	{59904, 14976, 4, 384, 39},
	{60160, 15040, 4, 320, 47},
	{60416, 15104, 4, 256, 59},
	{60480, 12096, 5, 448, 27},
	{60800, 12160, 5, 320, 38},
	{60928, 15232, 4, 448, 34},
	{61440, 15360, 4, 512, 30},
	{61824, 10304, 6, 448, 23},
	{62208, 10368, 6, 384, 27},
	{62400, 12480, 5, 320, 39},
	{62464, 15616, 4, 256, 61},
	{62720, 15680, 4, 448, 35},
	{62976, 15744, 4, 384, 41},
	{63360, 12672, 5, 384, 33},
	{63488, 15872, 4, 512, 31},
	{64000, 16000, 4, 320, 50},
	{64512, 16128, 4, 448, 36},
	{64960, 12992, 5, 448, 29},
	{65280, 16320, 4, 320, 51},
	{65536, 16384, 4, 512, 32}
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
					{ cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ldc, lda, ldb, 1.0, aip, lda, bip, ldb, 1.0, cip, ldc); }
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
			return j; // b[0, j+1]
		else
			if(i < nB -1)
				return j; // b[0, 0]
			else
				return -1;
	}
}

void do_work()
{
	int ai, bi, oldai, oldbi;
	void *abaseptr, *bbaseptr;
	struct aml_scratch_request *ar, *br;
	abaseptr = aml_scratch_baseptr(&sa);
	bbaseptr = aml_scratch_baseptr(&sb);
	ai = -1; bi = -1;
	aml_scratch_pull(&sa, abaseptr, &ai, a, 0);
	aml_scratch_pull(&sb, bbaseptr, &bi, b, 0);
	for(int i = 0; i < nB; i++)
	{
		for(int j = 0; j < nB; j++)
		{
			for(int k = 0; k < nB; k++)
			{
				double *ap, *bp, *cp;
				int nai, nbi;
				oldai = ai;
				oldbi = bi;
				nai = nextai(i,j,k);
				nbi = nextbi(i,j,k);
				if(nai != -1) aml_scratch_async_pull(&sa, &ar, abaseptr, &ai, a, nai);
				if(nbi != -1) aml_scratch_async_pull(&sb, &br, bbaseptr, &bi, b, nbi);
				ap = aml_tiling_tilestart(&tiling_prefetch, abaseptr, oldai);
				bp = aml_tiling_tilestart(&tiling_prefetch, bbaseptr, oldbi);
				cp = aml_tiling_tilestart(&tiling_prefetch, c, i*nB + j);
				do_innerwork(ap, bp, cp);
				if(nai != -1) aml_scratch_wait(&sa, ar);
				if(nbi != -1) aml_scratch_wait(&sb, br);
				aml_scratch_release(&sa, oldai);
				aml_scratch_release(&sb, oldbi);
			}
		}
	}
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
			return;
		}
	}
	assert(0);
}

int main(int argc, char* argv[])
{
	AML_ARENA_JEMALLOC_DECL(arns);
	AML_ARENA_JEMALLOC_DECL(arnf);
	AML_DMA_LINUX_SEQ_DECL(dma);
	struct bitmask *slowb, *fastb;
	aml_init(&argc, &argv);
	assert(argc == 4);
	fastb = numa_parse_nodestring_all(argv[1]);
	slowb = numa_parse_nodestring_all(argv[2]);
	N = atol(argv[3]);
        find_blocking_tiling();
	assert(N % T == 0);
	memsize = sizeof(double)*N*N;
	tilesize = sizeof(double)*T*T;
	blocksize = sizeof(double)*B*B;

	/* A inner tiling of each block, and a tiling of blocks  */
	assert(!aml_tiling_init(&tiling_inner, AML_TILING_TYPE_2D_CONTIG_ROWMAJOR,
				tilesize, blocksize, nT , nT));
	assert(!aml_tiling_init(&tiling_prefetch, AML_TILING_TYPE_2D_CONTIG_ROWMAJOR,
				blocksize, memsize, nB, nB));

	assert(!aml_arena_jemalloc_init(&arns, AML_ARENA_JEMALLOC_TYPE_ALIGNED, (size_t)(64)));
	assert(!aml_area_linux_init(&slow,
				    AML_AREA_LINUX_MANAGER_TYPE_SINGLE,
				    AML_AREA_LINUX_MBIND_TYPE_REGULAR,
				    AML_AREA_LINUX_MMAP_TYPE_ANONYMOUS,
				    &arns, MPOL_BIND, slowb->maskp));
	assert(!aml_arena_jemalloc_init(&arnf, AML_ARENA_JEMALLOC_TYPE_ALIGNED, (size_t)(64)));
	assert(!aml_area_linux_init(&fast,
				    AML_AREA_LINUX_MANAGER_TYPE_SINGLE,
				    AML_AREA_LINUX_MBIND_TYPE_REGULAR,
				    AML_AREA_LINUX_MMAP_TYPE_ANONYMOUS,
				    &arnf, MPOL_BIND, fastb->maskp));
	assert(!aml_dma_linux_seq_init(&dma, 2));
	assert(!aml_scratch_par_init(&sa, &fast, &slow, &dma, &tiling_prefetch, (size_t)2, (size_t)2));
	assert(!aml_scratch_par_init(&sb, &fast, &slow, &dma, &tiling_prefetch, (size_t)2, (size_t)2));
	/* allocation */
	a = aml_area_malloc(&slow, memsize);
	b = aml_area_malloc(&slow, memsize);
	c = aml_area_malloc(&slow, memsize);
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
	aml_dma_linux_seq_destroy(&dma);
	aml_area_free(&slow, a);
	aml_area_free(&slow, b);
	aml_area_free(&fast, c);
	aml_area_linux_destroy(&slow);
	aml_area_linux_destroy(&fast);
	aml_tiling_destroy(&tiling_inner, AML_TILING_TYPE_2D_CONTIG_ROWMAJOR);
	aml_tiling_destroy(&tiling_prefetch, AML_TILING_TYPE_2D_CONTIG_ROWMAJOR);
	aml_finalize();
	return 0;
}
