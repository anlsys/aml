#include<time.h>
#include<aml.h>
#include<aml/dma/ze.h>
#include<level_zero/ze_api.h>

static int64_t timediff(struct timespec end, struct timespec start)
{
	int64_t e = end.tv_sec * 1e9 + end.tv_nsec;
	int64_t s = s.tv_sec * 1e9 + s.tv_nsec;
	return e - s;
}

static double bw(size_t size, int64_t time)
{
	return (1e-3 * size) / time;
}


int do_copy_1wait(struct aml_dma *dma, void *dest, void *src, size_t size, int repeat)
{
	int64_t mintime = UINT64_MAX;
	struct timespec start, end;
	for(int i = 0; i < repeat; i++) {
		clock_gettime(CLOCK_MONOTONIC, &start);
		aml_dma_copy_custom(dma, dest, src, aml_dma_ze_memcpy_op, size);
		clock_gettime(CLOCK_MONOTONIC, &end);
		mintime = min(mintime, timediff(end, start));
	}
	fprintf(stdout, "BW: %gGBytes/s\n", bw(size, mintime));
	return 0;
}

int do_copy_chunks(struct aml_dma *dma, void *dest, void *src, size_t size, size_t chunks, int repeat)
{
	int64_t mintime = UINT64_MAX;
	struct timespec start, end;
	size_t cksz = size/chunks;
	for(int i = 0; i < repeat; i++) {
		clock_gettime(CLOCK_MONOTONIC, &start);
		for(int j = 0; j < chunks; j++) {
			aml_dma_async_copy_custom(dma, NULL, dest+(j*cks), src + (j*cks), aml_dma_ze_memcpy_op, cksz);
		}
		aml_dma_barrier(dma);
		clock_gettime(CLOCK_MONOTONIC, &end);
		mintime = min(mintime, timediff(end, start));
	}
	fprintf(stdout, "BW: %gGBytes/s\n", bw(size, mintime));
	return 0;
}

int do_copy_2barrier(struct aml_dma *dma, void *dest1, void *src1,
		void *dest2, void *src2, size_t size, int repeat)
{
	int64_t mintime = UINT64_MAX;
	struct timespec start, end;
	for(int i = 0; i < repeat; i++) {
		clock_gettime(CLOCK_MONOTONIC, &start);
		aml_dma_async_copy_custom(dma, NULL, dest1, src1, aml_dma_ze_memcpy_op, size);
		aml_dma_async_copy_custom(dma, NULL, dest2, src2, aml_dma_ze_memcpy_op, size);
		aml_dma_barrier(dma);
		clock_gettime(CLOCK_MONOTONIC, &end);
		mintime = min(mintime, timediff(end, start));
	}
	fprintf(stdout, "BW: %gGBytes/s\n", bw(size, mintime));
	return 0;
}

int do_copy_2chunks(struct aml_dma *dma, void *dest1, void *src1,
		void *dest2, void *src2, size_t size, size_t chunks, int repeat)
{
	int64_t mintime = UINT64_MAX;
	struct timespec start, end;
	size_t cksz = size/chunks;
	for(int i = 0; i < repeat; i++) {
		clock_gettime(CLOCK_MONOTONIC, &start);
		for(int j = 0; j < chunks; j++) {
			aml_dma_async_copy_custom(dma, NULL, dest1+(j*cks), src1 + (j*cks), aml_dma_ze_memcpy_op, cksz);
			aml_dma_async_copy_custom(dma, NULL, dest2+(j*cks), src2 + (j*cks), aml_dma_ze_memcpy_op, cksz);
		}
		aml_dma_barrier(dma);
		clock_gettime(CLOCK_MONOTONIC, &end);
		mintime = min(mintime, timediff(end, start));
	}
	fprintf(stdout, "BW: %gGBytes/s\n", bw(size, mintime));
	return 0;
}

int main(int argc, char *argv[])
{
	const size_t bytes = 1e9;
	const size_t N = bytes / sizeof(double);
	const int repeats = 100;
	
	aml_init(&argc, &argv);

	double *host = aml_area_mmap(aml_area_ze_host, N * sizeof(double), NULL);
	double *device = aml_area_mmap(aml_area_ze_device, N * sizeof(double), NULL);

	fprintf("Default ZE DMA\n");
	fprintf("OMP: H2D: 1C-wait\n");
	do_copy_1way(aml_dma_ze_default, device, host, bytes, repeats);
	
	for(size_t i = 1; i < 20; i++) {
		fprintf("Default ZE DMA\n");
		fprintf("OMP: H2D: Chunks: %zu\n", i);
		do_copy_chunks(aml_dma_ze_default, device, host, bytes, i, repeats);
	}

	fprintf("Default ZE DMA\n");
	fprintf("OMP: D2H: 1C-wait\n");
	do_copy_1way(aml_dma_ze_default, host, device, bytes, repeats);
	
	for(size_t i = 1; i < 100; i++) {
		fprintf("Default ZE DMA\n");
		fprintf("OMP: H2D: Chunks: %zu\n", i);
		do_copy_chunks(aml_dma_ze_default, host, device, bytes, i, repeats);
	}

	double *host2= aml_area_mmap(aml_area_ze_host, N * sizeof(double), NULL);
	double *device2= aml_area_mmap(aml_area_ze_device, N * sizeof(double), NULL);

	fprintf("Default ZE DMA\n");
	fprintf("OMP: 2Way: 1C-barrier\n");
	do_copy_2barrier(aml_dma_ze_default, host, device, device2, host2, bytes, repeats);
	
	for(size_t i = 1; i < 100; i++) {
		fprintf("Default ZE DMA\n");
		fprintf("OMP: 2Way: Chunks: %zu\n", i);
		do_copy_2chunks(aml_dma_ze_default, host, device, device2, host2, bytes, i, repeats);
	}
	aml_finalize();
	return 1;
}

